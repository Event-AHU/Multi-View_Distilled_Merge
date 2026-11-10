
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class NLLLoss(BaseWeightedLoss):
    

    def _forward(self, cls_score, label, **kwargs):
        
        loss_cls = F.nll_loss(cls_score, label, **kwargs)
        return loss_cls
