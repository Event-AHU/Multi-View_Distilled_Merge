
from typing import Optional

import torch.nn as nn
from torch import Tensor

from mmaction.registry import MODELS
from .tsn_head import TSNHead


@MODELS.register_module()
class TPNHead(TSNHead):
    

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.spatial_type == 'avg':

            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        self.new_cls = None

    def _init_new_cls(self) -> None:
        self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0)
        if next(self.fc_cls.parameters()).is_cuda:
            self.new_cls = self.new_cls.cuda()
        self.new_cls.weight.copy_(self.fc_cls.weight[..., None, None, None])
        self.new_cls.bias.copy_(self.fc_cls.bias)

    def forward(self,
                x,
                num_segs: Optional[int] = None,
                fcn_test: bool = False,
                **kwargs) -> Tensor:
        
        if fcn_test:
            if self.avg_pool3d:
                x = self.avg_pool3d(x)
            if self.new_cls is None:
                self._init_new_cls()
            cls_score_feat_map = self.new_cls(x)
            return cls_score_feat_map

        if self.avg_pool2d is None:
            kernel_size = (1, x.shape[-2], x.shape[-1])
            self.avg_pool2d = nn.AvgPool3d(kernel_size, stride=1, padding=0)

        if num_segs is None:

            x = self.avg_pool3d(x)
        else:

            x = self.avg_pool2d(x)

            x = x.reshape((-1, num_segs) + x.shape[1:])

            x = self.consensus(x)

            x = x.squeeze(1)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)

        cls_score = self.fc_cls(x)

        return cls_score
