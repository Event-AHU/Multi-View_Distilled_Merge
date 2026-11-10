
from numbers import Number
from typing import Sequence, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData, LabelData
from mmengine.utils import is_str


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence,
                              int]) -> torch.Tensor:
    


    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1,\
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: Union[torch.Tensor, np.ndarray,
                              Sequence]) -> torch.Tensor:
    

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1,\
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


class ActionDataSample(BaseDataElement):

    def set_gt_labels(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ActionDataSample':
        
        label_data = getattr(self, '_gt_label', LabelData())
        label_data.item = format_label(value)
        self.gt_labels = label_data
        return self

    def set_pred_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ActionDataSample':
        
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.item = format_label(value)
        self.pred_labels = label_data
        return self

    def set_pred_score(self, value: torch.Tensor) -> 'ActionDataSample':
        
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.item = format_score(value)
        if hasattr(self, 'num_classes'):
            assert len(label_data.item) == self.num_classes,\
                f'The length of score {len(label_data.item)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes',
                value=len(label_data.item),
                field_type='metainfo')
        self.pred_scores = label_data
        return self

    @property
    def gt_labels(self):
        
        return self._gt_labels

    @gt_labels.setter
    def gt_labels(self, value):
        
        self.set_field(value, '_gt_labels', LabelData)

    @gt_labels.deleter
    def gt_labels(self):
        
        del self._gt_labels

    @property
    def pred_scores(self):
        
        return self._pred_scores

    @pred_scores.setter
    def pred_scores(self, value):
        
        self.set_field(value, '_pred_scores', LabelData)

    @pred_scores.deleter
    def pred_scores(self):
        
        del self._pred_scores

    @property
    def pred_labels(self):
        
        return self._pred_labels

    @pred_labels.setter
    def pred_labels(self, value):
        
        self.set_field(value, '_pred_labels', LabelData)

    @pred_labels.deleter
    def pred_labels(self):
        
        del self._pred_labels

    @property
    def proposals(self):
        
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        
        del self._proposals

    @property
    def gt_instances(self):
        
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        
        del self._gt_instances
