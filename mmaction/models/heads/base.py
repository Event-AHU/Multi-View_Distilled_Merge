
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import LabelData

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList


class AvgConsensus(nn.Module):
    

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(BaseModule, metaclass=ABCMeta):
    

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: Dict = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 average_clips: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super(BaseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def forward(self, x, **kwargs) -> ForwardResults:
        
        raise NotImplementedError

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes\
                and cls_scores.size()[0] == 1:



            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)

        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            prediction = LabelData(item=score)
            pred_label = LabelData(item=pred_label)
            data_sample.pred_scores = prediction
            data_sample.pred_labels = pred_label
        return data_samples

    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:
        

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view(batch_size // num_segs, num_segs, -1)

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores
