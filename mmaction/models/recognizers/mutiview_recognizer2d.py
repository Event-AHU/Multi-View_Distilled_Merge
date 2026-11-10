import torch.nn as nn
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseRecognizer


@MODELS.register_module()
class MutiviewRecognizer2D(BaseRecognizer):
    

    @staticmethod
    def _squeeze_clip_dim(x: Tensor) -> Tensor:
        if x is not None and x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)
        return x

    @staticmethod
    def _ensure_rgb_ntchw(x: Tensor) -> Tensor:
        x = MutiviewRecognizer2D._squeeze_clip_dim(x)
        if x.dim() != 5:
            raise RuntimeError(f'RGB tensor must be [B,T,C,H,W], got shape={tuple(x.shape)}')
        if x.size(2) != 3 and x.size(-1) == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if x.size(2) != 3:
            raise RuntimeError(f'RGB tensor channel dim must be 3, got shape={tuple(x.shape)}')
        return x

    @staticmethod
    def _ensure_event_ntchw(x: Tensor) -> Tensor:
        x = MutiviewRecognizer2D._squeeze_clip_dim(x)
        if x.dim() != 5:
            raise RuntimeError(
                f'Event tensor must be [B,T,C,H,W] or [B,T,H,W,C], got shape={tuple(x.shape)}')
        if x.size(2) in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            return x
        if x.size(-1) in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            return x.permute(0, 1, 4, 2, 3).contiguous()
        raise RuntimeError(f'Cannot infer event channel dim from shape={tuple(x.shape)}')

    def extract_feat(self,
                     inputs: Tensor,
                     imgs_event: Tensor,
                     stage: str = 'neck',
                     data_samples: SampleList = None,
                     test_mode: bool = False) -> tuple:
        loss_predict_kwargs = dict()

        if imgs_event is None:

            inputs = self._ensure_rgb_ntchw(inputs)
            total_segs = inputs.shape[1]
            if total_segs % 2 != 0:
                raise RuntimeError(
                    f'Fallback concatenated input expects even T, got T={total_segs}')
            num_segs = total_segs // 2
            inputs_rgb = inputs[:, :num_segs, :, :, :]
            inputs_event = inputs[:, num_segs:, :, :, :]
        else:
            inputs_rgb = self._ensure_rgb_ntchw(inputs)
            inputs_event = self._ensure_event_ntchw(imgs_event)
            num_segs = inputs_rgb.shape[1]
            if inputs_event.shape[1] != num_segs:
                raise RuntimeError(
                    f'RGB/Event temporal length mismatch: {inputs_rgb.shape[1]} vs {inputs_event.shape[1]}')



        loss_predict_kwargs['num_segs'] = getattr(self.cls_head, 'num_segments', num_segs * 2)

        fcn_test = False
        if test_mode and self.test_cfg is not None and self.test_cfg.get('fcn_test', False):
            fcn_test = True
            loss_predict_kwargs['fcn_test'] = fcn_test

        x = self.backbone([inputs_rgb, inputs_event], num_segs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 3 and x.shape[2] > 1:
                x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2))
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if stage == 'backbone':
            return x, loss_predict_kwargs

        loss_aux = dict()
        if self.with_neck:
            total_segments = getattr(self.cls_head, 'num_segments', num_segs * 2)
            x = [
                each.reshape((-1, total_segments) + each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, data_samples=data_samples)
            if not fcn_test:
                x = x.squeeze(2)
                loss_predict_kwargs['num_segs'] = 1
        elif fcn_test:
            total_segments = getattr(self.cls_head, 'num_segments', num_segs * 2)
            x = x.reshape((-1, total_segments) + x.shape[1:]).transpose(1, 2).contiguous()

        loss_predict_kwargs['loss_aux'] = loss_aux
        if stage == 'neck':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs

        return x, loss_predict_kwargs
