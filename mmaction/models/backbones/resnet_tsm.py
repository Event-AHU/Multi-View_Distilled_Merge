
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, NonLocal3d
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from torch.nn.modules.utils import _ntuple

from mmaction.registry import MODELS
from .resnet import ResNet


class NL3DWrapper(nn.Module):
    

    def __init__(self, block, num_segments, non_local_cfg=dict()):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.non_local_cfg = non_local_cfg
        self.non_local_block = NonLocal3d(self.block.conv3.norm.num_features,
                                          **self.non_local_cfg)
        self.num_segments = num_segments

    def forward(self, x):
        
        x = self.block(x)

        n, c, h, w = x.size()
        x = x.view(n // self.num_segments, self.num_segments, c, h,
                   w).transpose(1, 2).contiguous()
        x = self.non_local_block(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        return x


class TemporalShift(nn.Module):
    

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        

        n, c, h, w = x.size()



        x = x.view(-1, num_segments, c, h * w)


        fold = c // shift_div



        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]





        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)


        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)




        out = torch.cat((left_split, mid_split, right_split), 2)



        return out.view(n, c, h, w)


@MODELS.register_module()
class ResNetTSM(ResNet):
    

    def __init__(self,
                 depth,
                 num_segments=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 pretrained2d=True,
                 **kwargs):
        super().__init__(depth, **kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.non_local_stages = _ntuple(self.num_stages)(non_local)
        self.non_local_cfg = non_local_cfg
        self.pretrained2d = pretrained2d
        self.init_structure()

    def init_structure(self):
        
        if self.is_shift:
            self.make_temporal_shift()
        if len(self.non_local_cfg) != 0:
            self.make_non_local()
        if self.temporal_pool:
            self.make_temporal_pool()

    def make_temporal_shift(self):
        
        if self.temporal_pool:
            num_segment_list = [
                self.num_segments, self.num_segments // 2,
                self.num_segments // 2, self.num_segments // 2
            ]
        else:
            num_segment_list = [self.num_segments] * 4
        if num_segment_list[-1] <= 0:
            raise ValueError('num_segment_list[-1] must be positive')

        if self.shift_place == 'block':

            def make_block_temporal(stage, num_segments):
                
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(
                        b, num_segments=num_segments, shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        elif 'blockres' in self.shift_place:
            n_round = 1
            if len(list(self.layer3.children())) >= 23:
                n_round = 2

            def make_block_temporal(stage, num_segments):
                
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1.conv = TemporalShift(
                            b.conv1.conv,
                            num_segments=num_segments,
                            shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        else:
            raise NotImplementedError

    def make_temporal_pool(self):
        

        class TemporalPool(nn.Module):
            

            def __init__(self, net, num_segments):
                super().__init__()
                self.net = net
                self.num_segments = num_segments
                self.max_pool3d = nn.MaxPool3d(
                    kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

            def forward(self, x):
                

                n, c, h, w = x.size()

                x = x.view(n // self.num_segments, self.num_segments, c, h,
                           w).transpose(1, 2)

                x = self.max_pool3d(x)

                x = x.transpose(1, 2).contiguous().view(n // 2, c, h, w)
                return self.net(x)

        self.layer2 = TemporalPool(self.layer2, self.num_segments)

    def make_non_local(self):
        

        for i in range(self.num_stages):
            non_local_stage = self.non_local_stages[i]
            if sum(non_local_stage) == 0:
                continue

            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)

            for idx, non_local in enumerate(non_local_stage):
                if non_local:
                    res_layer[idx] = NL3DWrapper(res_layer[idx],
                                                 self.num_segments,
                                                 self.non_local_cfg)

    def load_original_weights(self, logger):
        
        state_dict_torchvision = _load_checkpoint(
            self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        wrapped_layers_map = dict()
        for name, module in self.named_modules():

            ori_name = name
            for wrap_prefix in ['.net', '.block']:
                if wrap_prefix in ori_name:
                    ori_name = ori_name.replace(wrap_prefix, '')
                    wrapped_layers_map[ori_name] = name

            if isinstance(module, ConvModule):
                if 'downsample' in ori_name:

                    tv_conv_name = ori_name + '.0'

                    tv_bn_name = ori_name + '.1'
                else:

                    tv_conv_name = ori_name

                    tv_bn_name = ori_name.replace('conv', 'bn')

                for conv_param in ['.weight', '.bias']:
                    if tv_conv_name + conv_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name+'.conv'+conv_param] =\
                            state_dict_torchvision.pop(tv_conv_name+conv_param)

                for bn_param in [
                        '.weight', '.bias', '.running_mean', '.running_var'
                ]:
                    if tv_bn_name + bn_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name+'.bn'+bn_param] =\
                            state_dict_torchvision.pop(tv_bn_name+bn_param)


        for param_name in list(state_dict_torchvision.keys()):
            layer_name = '.'.join(param_name.split('.')[:-1])
            if layer_name in wrapped_layers_map:
                wrapped_name = param_name.replace(
                    layer_name, wrapped_layers_map[layer_name])
                state_dict_torchvision[
                    wrapped_name] = state_dict_torchvision.pop(param_name)

        msg = self.load_state_dict(state_dict_torchvision, strict=False)
        logger.info(msg)

    def init_weights(self):
        
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            self.load_original_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()
