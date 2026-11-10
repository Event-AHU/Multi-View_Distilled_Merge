
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class SingleRoIExtractor3D(nn.Module):
    

    def __init__(self,
                 roi_layer_type: str = 'RoIAlign',
                 featmap_stride: int = 16,
                 output_size: int = 16,
                 sampling_ratio: int = 0,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 with_temporal_pool: bool = True,
                 temporal_pool_mode: str = 'avg',
                 with_global: bool = False) -> None:
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        try:
            from mmcv.ops import RoIAlign, RoIPool
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `RoIAlign` and `RoIPool` from '
                              '`mmcv.ops`. The two modules will be used in '
                              '`SingleRoIExtractor3D`! ')

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, feat: Union[Tensor, Tuple[Tensor]],
                rois: Tensor) -> tuple:
        
        if not isinstance(feat, tuple):
            feat = (feat, )

        if len(feat) >= 2:
            maxT = max([x.shape[2] for x in feat])
            max_shape = (maxT, ) + feat[0].shape[3:]

            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            roi_feat = self.roi_layer(frame_feat, rois)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        roi_feats = torch.stack(roi_feats, dim=2)
        return roi_feats, feat


if mmdet_imported:
    MMDET_MODELS.register_module()(SingleRoIExtractor3D)
