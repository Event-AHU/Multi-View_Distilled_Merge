
from typing import List, Tuple

from mmengine.model.weight_init import constant_init, trunc_normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class MViTHead(BaseHead):
    

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.02,
                 init_scale: float = 1.0,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.init_scale = init_scale
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        
        trunc_normal_init(self.fc_cls.weight, std=self.init_std)
        constant_init(self.fc_cls.bias, 0.02)
        self.fc_cls.weight.data.mul_(self.init_scale)
        self.fc_cls.bias.data.mul_(self.init_scale)

    def pre_logits(self, feats: Tuple[List[Tensor]]) -> Tensor:
        
        _, cls_token = feats[-1]
        return cls_token

    def forward(self, x: Tuple[List[Tensor]], **kwargs) -> Tensor:
        
        x = self.pre_logits(x)
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)

        return cls_score
