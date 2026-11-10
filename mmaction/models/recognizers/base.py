
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)


class BaseRecognizer(BaseModel, metaclass=ABCMeta):
    

    def __init__(self,
                 backbone: ConfigType,
                 cls_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        if data_preprocessor is None:

            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(BaseRecognizer,self).__init__(data_preprocessor=data_preprocessor)


        self.backbone_from = 'mmaction2'

        if backbone['type'].startswith('mmcls.'):
            try:

                import mmcls.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this backbone.')
            backbone_type = backbone.pop('type')[12:]
            self.backbone = torchvision.models.__dict__[backbone_type](**backbone)

            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[5:]

            backbone['num_classes'] = 0
            self.backbone = timm.create_model(backbone_type, **backbone)
            self.backbone_from = 'timm'
        else:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def extract_feat(self, inputs: torch.Tensor, imgs_event: torch.Tensor,**kwargs) -> ForwardResults:
        pass

    @property
    def with_neck(self) -> bool:
        
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self) -> None:
        
        super().init_weights()
        if self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized '
                          'in their __init__ functions.')

    def loss(self, inputs: torch.Tensor, imgs_event: torch.Tensor,data_samples: SampleList, **kwargs) -> dict:
        

        feats, loss_kwargs = self.extract_feat(inputs=inputs, imgs_event=imgs_event, data_samples=data_samples)


        loss_aux = loss_kwargs.get('loss_aux', dict())
        loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)
        losses = merge_dict(loss_cls, loss_aux)
        return losses

    def predict(self, inputs: torch.Tensor, imgs_event: torch.Tensor,data_samples: SampleList, **kwargs) -> SampleList:
        
        feats, predict_kwargs = self.extract_feat(inputs, imgs_event, test_mode=True)
        predictions = self.cls_head.predict(feats, data_samples, **predict_kwargs)
        return predictions

    def _forward(self,
                 inputs: torch.Tensor,
                 imgs_event: torch.Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        
        feats, _ = self.extract_feat(inputs, imgs_event, stage=stage)
        return feats

    def forward(self,
                inputs: torch.Tensor,
                imgs_event: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        
        if mode == 'tensor':
            return self._forward(inputs, imgs_event, **kwargs)
        if mode == 'predict':
            return self.predict(inputs, imgs_event, data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(inputs, imgs_event, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
