
import torch.nn as nn
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer2D(BaseRecognizer):
    

    def extract_feat(self,
                     inputs: Tensor,
                     imgs_event: Tensor,
                     stage: str = 'neck',
                     data_samples: SampleList = None,
                     test_mode: bool = False) -> tuple:
        


        loss_predict_kwargs = dict()


        num_segs = inputs.shape[1]
        loss_predict_kwargs['num_segs'] = num_segs










        inputs_rgb   = inputs[:, :num_segs//2, :, :, :]
        inputs_event = inputs[:, num_segs//2:, :, :, :]
        inputs = [inputs_rgb, inputs_event]


        fcn_test = False
        if test_mode:
            if self.test_cfg is not None and self.test_cfg.get(
                    'fcn_test', False):
                fcn_test = True
                num_segs = self.test_cfg.get('num_segs',
                                             self.backbone.num_segments)
            loss_predict_kwargs['fcn_test'] = fcn_test


        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            x = self.backbone.features(inputs)
        elif self.backbone_from == 'timm':
            x = self.backbone.forward_features(inputs)
        elif self.backbone_from == 'mmcls':
            x = self.backbone(inputs)
            if isinstance(x, tuple):
                assert len(x) == 1
                x = x[0]
        else:
            x = self.backbone(inputs, num_segs)

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

            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, data_samples=data_samples)
            if not fcn_test:
                x = x.squeeze(2)
                loss_predict_kwargs['num_segs'] = 1
        elif fcn_test:



            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        loss_predict_kwargs['loss_aux'] = loss_aux



        if stage == 'neck':
            return x, loss_predict_kwargs



        if self.with_cls_head and stage == 'head':


            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
