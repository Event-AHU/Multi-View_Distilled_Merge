
from typing import Optional

from mmengine.fileio import load
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class UniFormerHead(BaseHead):
    

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 dropout_ratio: float = 0.0,
                 channel_map: Optional[str] = None,
                 init_cfg: Optional[dict] = dict(
                     type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs) -> None:
        super().__init__(
            num_classes, in_channels, loss_cls, init_cfg=init_cfg, **kwargs)
        self.channel_map = channel_map
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def _select_channels(self, stact_dict):
        selected_channels = load(self.channel_map)
        for key in stact_dict:
            stact_dict[key] = stact_dict[key][selected_channels]

    def init_weights(self) -> None:
        
        if self.init_cfg['type'] == 'Pretrained':
            assert self.channel_map is not None,\
                'load cls_head weights needs to specify the channel map file'
            logger = MMLogger.get_current_instance()
            pretrained = self.init_cfg['checkpoint']
            logger.info(f'load pretrained model from {pretrained}')
            state_dict = _load_checkpoint_with_prefix(
                'cls_head.', pretrained, map_location='cpu')
            self._select_channels(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
        else:
            super().init_weights()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)

        return cls_score
