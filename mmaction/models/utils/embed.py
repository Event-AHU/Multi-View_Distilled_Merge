
import math

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.utils import to_3tuple


class AdaptivePadding(nn.Module):
    

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        
        input_t, input_h, input_w = input_shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        output_d = math.ceil(input_t / stride_d)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_d = max((output_d - 1) * stride_d +
                    (kernel_d - 1) * self.dilation[0] + 1 - input_t, 0)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[1] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[2] + 1 - input_w, 0)
        return pad_d, pad_h, pad_w

    def forward(self, x):
        
        pad_d, pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_d // 2,
                    pad_d - pad_d // 2,
                ])
        return x


class PatchEmbed3D(BaseModule):
    

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv3d',
                 kernel_size=(2, 4, 4),
                 stride=(2, 4, 4),
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)

            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_3tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_3tuple(input_size)



            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_d, pad_h, pad_w = self.adaptive_padding.get_pad_shape(
                    input_size)
                input_t, input_h, input_w = input_size
                input_t = input_t + pad_d
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_t, input_h, input_w)


            t_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            h_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            w_out = (input_size[2] + 2 * padding[2] - dilation[2] *
                     (kernel_size[2] - 1) - 1) // stride[2] + 1
            self.init_out_size = (t_out, h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3], x.shape[4])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
