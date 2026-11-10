import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmaction.registry import MODELS
from .resnet_tsm import ResNetTSM
from .utils.former import Former, Res2Former


@MODELS.register_module()
class MutiviewTSCFormer(BaseModule):
    

    def __init__(self,
                 depth,
                 pretrained='',
                 num_segments=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 pretrained2d=True,
                 dropout_ratio: float = 0.8,
                 event_in_channels: int = 3,
                 event_adapter_out_channels: int = 64,
                 init_event_adapter_from_resnet: bool = True,
                 **kwargs):
        super().__init__()
        self.num_segments = int(num_segments)
        self.event_in_channels = int(event_in_channels)
        self.event_adapter_out_channels = int(event_adapter_out_channels)
        self.init_event_adapter_from_resnet = bool(init_event_adapter_from_resnet)

        self.ResNetTSM = ResNetTSM(
            depth=depth,
            pretrained=pretrained,
            num_segments=num_segments,
            is_shift=is_shift,
            non_local=non_local,
            non_local_cfg=non_local_cfg,
            shift_div=shift_div,
            shift_place=shift_place,
            temporal_pool=temporal_pool,
            pretrained2d=pretrained2d)

        self.ResNetTSM_event = ResNetTSM(
            depth=depth,
            pretrained=pretrained,
            num_segments=num_segments,
            is_shift=is_shift,
            non_local=non_local,
            non_local_cfg=non_local_cfg,
            shift_div=shift_div,
            shift_place=shift_place,
            temporal_pool=temporal_pool,
            pretrained2d=pretrained2d)


        self.ResNetTSM.init_weights()
        self.ResNetTSM_event.init_weights()




        self.event_adapter = ConvModule(
            self.event_in_channels,
            self.event_adapter_out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True))

        if self.init_event_adapter_from_resnet:
            self._init_event_adapter_from_original_conv1()


        self.ResNetTSM_event.conv1 = nn.Identity()

        former_dim = 64
        self.former3 = Former(dim=former_dim, depth=1)
        self.Res2former3 = Res2Former(dim=former_dim, heads=2, channel=1024)
        self.former4 = Former(dim=former_dim, depth=1)
        self.Res2former4 = Res2Former(dim=former_dim, heads=2, channel=2048)
        token_num = 3
        self.token = nn.Parameter(torch.randn(1, token_num, former_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_res = nn.Dropout(p=dropout_ratio)

        self.fc_layer3 = nn.Linear(former_dim * token_num * 2, 1 * 14 * 14)
        self.fc_layer4 = nn.Linear(former_dim * token_num * 2, 1 * 7 * 7)
        self.conv_1x1_rgb_layer3 = nn.Conv2d(1025, 1024, kernel_size=1)
        self.conv_1x1_event_layer3 = nn.Conv2d(1025, 1024, kernel_size=1)
        self.conv_1x1_rgb_layer4 = nn.Conv2d(2049, 2048, kernel_size=1)
        self.conv_1x1_event_layer4 = nn.Conv2d(2049, 2048, kernel_size=1)

    def _init_event_adapter_from_original_conv1(self):
        
        try:
            src_conv = self.ResNetTSM_event.conv1.conv
            dst_conv = self.event_adapter.conv

            w = src_conv.weight.data
            if self.event_in_channels == w.shape[1]:
                w_new = w.clone()
            else:
                w_mean = w.mean(dim=1, keepdim=True)
                w_new = w_mean.repeat(1, self.event_in_channels, 1, 1)
                w_new *= (float(w.shape[1]) / float(self.event_in_channels))

            if dst_conv.weight.data.shape == w_new.shape:
                dst_conv.weight.data.copy_(w_new)


            src_bn = getattr(self.ResNetTSM_event.conv1, 'bn', None)
            if src_bn is None:
                src_bn = getattr(self.ResNetTSM_event.conv1, 'norm', None)
            dst_bn = getattr(self.event_adapter, 'bn', None)
            if dst_bn is None:
                dst_bn = getattr(self.event_adapter, 'norm', None)

            if src_bn is not None and dst_bn is not None:
                dst_bn.weight.data.copy_(src_bn.weight.data)
                dst_bn.bias.data.copy_(src_bn.bias.data)
                dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
                dst_bn.running_var.data.copy_(src_bn.running_var.data)
                if hasattr(dst_bn, 'num_batches_tracked') and hasattr(src_bn, 'num_batches_tracked'):
                    dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)
        except Exception:

            pass

    def _ensure_rgb_ntchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 5:
            raise RuntimeError(f'RGB input must be 5D [B,T,C,H,W], got shape={tuple(x.shape)}')
        if x.size(2) != 3 and x.size(-1) == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if x.size(2) != 3:
            raise RuntimeError(f'RGB channel dim must be 3, got shape={tuple(x.shape)}')
        return x

    def _ensure_event_ntchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 5:
            raise RuntimeError(
                f'Event input must be 5D [B,T,C,H,W] or [B,T,H,W,C], got shape={tuple(x.shape)}')

        if x.size(2) == self.event_in_channels:
            pass
        elif x.size(-1) == self.event_in_channels:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        else:
            raise RuntimeError(
                f'Event channel mismatch: expected {self.event_in_channels}, got shape={tuple(x.shape)}')
        return x

    def forward(self, x: torch.Tensor, num_seg=None):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError('MutiviewTSCFormer expects [rgb_tensor, event_tensor].')

        num_seg = self.num_segments if num_seg is None else int(num_seg)
        inputs_rgb = self._ensure_rgb_ntchw(x[0])
        inputs_event = self._ensure_event_ntchw(x[1])

        x_rgb = inputs_rgb.contiguous().view((-1,) + inputs_rgb.shape[2:])
        x_event = inputs_event.contiguous().view((-1,) + inputs_event.shape[2:])

        x_rgb_stem_conv1 = self.ResNetTSM.conv1(x_rgb)
        x_rgb_stem = self.ResNetTSM.maxpool(x_rgb_stem_conv1)

        x_event_stem_conv1 = self.event_adapter(x_event)
        x_event_stem = self.ResNetTSM_event.maxpool(x_event_stem_conv1)

        out_rgb_layer1 = self.ResNetTSM.layer1(x_rgb_stem)
        out_rgb_layer2 = self.ResNetTSM.layer2(out_rgb_layer1)
        out_rgb_layer3 = self.ResNetTSM.layer3(out_rgb_layer2)

        out_event_layer1 = self.ResNetTSM_event.layer1(x_event_stem)
        out_event_layer2 = self.ResNetTSM_event.layer2(out_event_layer1)
        out_event_layer3 = self.ResNetTSM_event.layer3(out_event_layer2)

        nt_rgb, c, h, w = out_rgb_layer3.shape
        nt_event = out_event_layer3.shape[0]
        if nt_rgb != nt_event:
            raise RuntimeError(
                f'RGB/Event temporal batches mismatch at layer3: nt_rgb={nt_rgb}, nt_event={nt_event}')
        nt = nt_rgb + nt_event

        reshape_rgb_layer3 = out_rgb_layer3.view(-1, num_seg, c, h, w)
        reshape_event_layer3 = out_event_layer3.view(-1, num_seg, c, h, w)

        out_res_layer3 = torch.cat((reshape_rgb_layer3, reshape_event_layer3), 1)
        out_res_layer3 = out_res_layer3.view(-1, c, h, w)

        z = self.token.repeat(nt, 1, 1)
        z_hid3 = self.Res2former3(out_res_layer3, z)
        z_out3 = self.former3(z_hid3)

        z_out_layer3 = z_out3.view(nt_rgb, -1)
        z_out_layer3 = self.fc_layer3(z_out_layer3)
        z_out_layer3 = z_out_layer3.view(nt_rgb, -1, h, w)

        input_rgb_layer4 = torch.cat((out_rgb_layer3, z_out_layer3), 1)
        input_event_layer4 = torch.cat((out_event_layer3, z_out_layer3), 1)

        input_rgb_layer4 = self.conv_1x1_rgb_layer3(input_rgb_layer4)
        input_event_layer4 = self.conv_1x1_event_layer3(input_event_layer4)

        out_rgb_layer4 = self.ResNetTSM.layer4(input_rgb_layer4)
        out_event_layer4 = self.ResNetTSM_event.layer4(input_event_layer4)

        _, c_4, h_4, w_4 = out_rgb_layer4.shape
        reshape_rgb_layer4 = out_rgb_layer4.view(-1, num_seg, c_4, h_4, w_4)
        reshape_event_layer4 = out_event_layer4.view(-1, num_seg, c_4, h_4, w_4)

        out_res_layer4 = torch.cat((reshape_rgb_layer4, reshape_event_layer4), 1)
        out_res_layer4 = out_res_layer4.view(-1, c_4, h_4, w_4)

        z_hid4 = self.Res2former4(out_res_layer4, z_out3)
        z_out4 = self.former4(z_hid4)

        z_out_layer4 = z_out4.view(nt_rgb, -1)
        z_out_layer4 = self.fc_layer4(z_out_layer4)
        z_out_layer4 = z_out_layer4.view(nt_rgb, -1, h_4, w_4)

        out_r = torch.cat((out_rgb_layer4, z_out_layer4), 1)
        out_e = torch.cat((out_event_layer4, z_out_layer4), 1)

        out_r = self.conv_1x1_rgb_layer4(out_r)
        out_e = self.conv_1x1_event_layer4(out_e)

        out_r = out_r.view(-1, num_seg, c_4, h_4, w_4)
        out_e = out_e.view(-1, num_seg, c_4, h_4, w_4)

        out = torch.cat((out_r, out_e), 1)
        out = out.view(-1, c_4, h_4, w_4)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout_res(out)

        z_out = z_out4.view(nt, -1)
        out_all = torch.cat((out, z_out), 1)
        return out_all
