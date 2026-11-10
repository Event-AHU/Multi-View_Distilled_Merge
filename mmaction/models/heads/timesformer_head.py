
from mmengine.model.weight_init import trunc_normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class TimeSformerHead(BaseHead):
    

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 init_std: float = 0.02,
                 dropout_ratio: float = 0.0,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)

        return cls_score
