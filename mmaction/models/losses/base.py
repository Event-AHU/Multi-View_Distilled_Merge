
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        
        pass

    def forward(self, *args, **kwargs):
        
        ret = self._forward(*args, **kwargs)
        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            ret *= self.loss_weight
        return ret
