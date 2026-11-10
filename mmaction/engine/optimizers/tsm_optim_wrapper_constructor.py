
import torch
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm_, _BatchNorm, _ConvNd, LayerNorm)

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TSMOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    

    def add_params(self, params, model, **kwargs):
        


        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        ln = []

        conv_cnt = 0
        for m in model.modules():
            if isinstance(m, _ConvNd):
                m_params = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m_params[0])
                    if len(m_params) == 2:
                        first_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                normal_weight.append(m_params[0])
                if len(m_params) == 2:
                    normal_bias.append(m_params[1])
            elif isinstance(m,(_BatchNorm, SyncBatchNorm_, torch.nn.GroupNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif isinstance(m,(LayerNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        ln.append(param)
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')


        last_fc_weight = normal_weight.pop()
        last_fc_bias = normal_bias.pop()
        if fc_lr5:
            lr5_weight.append(last_fc_weight)
            lr10_bias.append(last_fc_bias)
        else:
            normal_weight.append(last_fc_weight)
            normal_bias.append(last_fc_bias)

        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': first_conv_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * 5,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': lr10_bias,
            'lr': self.base_lr * 10,
            'weight_decay': 0
        })
