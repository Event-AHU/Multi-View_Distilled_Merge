
from typing import Dict, Sequence, Union

import torch
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults, SampleList


@MODELS.register_module()
class RecognizerOmni(BaseModel):
    

    def __init__(self, backbone: ConfigType, cls_head: ConfigType,
                 data_preprocessor: ConfigType) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def forward(self, *data_samples, mode: str, **kwargs) -> ForwardResults:
        

        if mode == 'loss' or mode == 'predict':
            if mode == 'loss':
                return self.loss(data_samples)
            return self.predict(data_samples)

        elif mode == 'tensor':

            assert isinstance(data_samples, torch.Tensor)

            data_ndim = data_samples.ndim
            if data_ndim not in [4, 5]:
                info = f'Input is a {data_ndim}D tensor. '
                info += 'Only 4D (BCHW) or 5D (BCTHW) tensors are supported!'
                raise ValueError(info)

            return self._forward(data_samples, **kwargs)

    def loss(self, data_samples: Sequence[SampleList]) -> dict:
        
        loss_dict = {}
        for idx, data in enumerate(data_samples):
            inputs, data_samples = data['inputs'], data['data_samples']
            feats = self.extract_feat(inputs)
            loss_cls = self.cls_head.loss(feats, data_samples)
            for key in loss_cls:
                loss_dict[key + f'_{idx}'] = loss_cls[key]
        return loss_dict

    def predict(self, data_samples: Sequence[SampleList]) -> SampleList:
        
        assert len(data_samples) == 1
        feats = self.extract_feat(data_samples[0]['inputs'], test_mode=True)
        predictions = self.cls_head.predict(feats,
                                            data_samples[0]['data_samples'])
        return predictions

    def _forward(self,
                 inputs: torch.Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        
        feats, _ = self.extract_feat(inputs, stage=stage)
        return feats

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        
        if isinstance(data, dict):
            data = [data]
            results = self(*data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError
        return results

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = 'backbone',
                     test_mode: bool = False) -> tuple:
        

        if len(inputs.shape) == 6:
            inputs = inputs.view((-1, ) + inputs.shape[2:])


        if test_mode:
            x = self.backbone(inputs)
            return x
        else:

            x = self.backbone(inputs)
            if stage == 'backbone':
                return x
            x = self.cls_head(x)
            return x
