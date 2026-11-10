
from torch import Tensor

from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class RecognizerAudio(BaseRecognizer):
    

    def extract_feat(self,
                     batch_inputs: Tensor,
                     stage: str = 'backbone',
                     **kwargs) -> tuple:
        


        loss_predict_kwargs = dict()
        batch_inputs = batch_inputs.view((-1, ) + batch_inputs.shape[2:])

        x = self.backbone(batch_inputs)

        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
