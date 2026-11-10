
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from mmaction.registry import MODELS
from mmaction.utils import SampleList

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'CutmixBlending']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs: torch.Tensor, label: torch.Tensor,
                    **kwargs) -> Tuple:
        
        raise NotImplementedError

    def __call__(self, imgs: torch.Tensor, batch_data_samples: SampleList,
                 **kwargs) -> Tuple:
        
        label = [x.gt_labels.item for x in batch_data_samples]
        label = torch.tensor(label, dtype=torch.long).to(imgs.device)

        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        for label_item, sample in zip(mixed_label, batch_data_samples):
            sample.gt_labels.item = label_item

        return mixed_imgs, batch_data_samples


@MODELS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    

    def __init__(self, num_classes: int, alpha: float = .2) -> None:
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs: torch.Tensor, label: torch.Tensor,
                    **kwargs) -> Tuple:
        
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


@MODELS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    

    def __init__(self, num_classes: int, alpha: float = .2) -> None:
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size: torch.Size, lam: torch.Tensor) -> Tuple:
        
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))


        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs: torch.Tensor, label: torch.Tensor,
                    **kwargs) -> Tuple:
        

        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label


@MODELS.register_module()
class RandomBatchAugment(BaseMiniBatchBlending):
    

    def __init__(self,
                 augments: Union[dict, list],
                 probs: Optional[Union[float, List[float]]] = None) -> None:
        if not isinstance(augments, (tuple, list)):
            augments = [augments]

        self.augments = []
        for aug in augments:
            assert isinstance(aug, dict),\
                f'blending augment config must be a dict. Got {type(aug)}'
            self.augments.append(MODELS.build(aug))

        self.num_classes = augments[0].get('num_classes')

        if isinstance(probs, float):
            probs = [probs]

        if probs is not None:
            assert len(augments) == len(probs),\
                '``augments`` and ``probs`` must have same lengths. '\
                f'Got {len(augments)} vs {len(probs)}.'
            assert sum(probs) <= 1,\
                'The total probability of batch augments exceeds 1.'
            self.augments.append(None)
            probs.append(1 - sum(probs))

        self.probs = probs

    def do_blending(self, imgs: torch.Tensor, label: torch.Tensor,
                    **kwargs) -> Tuple:
        
        aug_index = np.random.choice(len(self.augments), p=self.probs)
        aug = self.augments[aug_index]

        if aug is not None:
            return aug.do_blending(imgs, label, **kwargs)
        else:
            return imgs, label
