
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class HVULoss(BaseWeightedLoss):
    

    def __init__(self,
                 categories=('action', 'attribute', 'concept', 'event',
                             'object', 'scene'),
                 category_nums=(739, 117, 291, 69, 1678, 248),
                 category_loss_weights=(1, 1, 1, 1, 1, 1),
                 loss_type='all',
                 with_mask=False,
                 reduction='mean',
                 loss_weight=1.0):

        super().__init__(loss_weight)
        self.categories = categories
        self.category_nums = category_nums
        self.category_loss_weights = category_loss_weights
        assert len(self.category_nums) == len(self.category_loss_weights)
        for category_loss_weight in self.category_loss_weights:
            assert category_loss_weight >= 0
        self.loss_type = loss_type
        self.with_mask = with_mask
        self.reduction = reduction
        self.category_startidx = [0]
        for i in range(len(self.category_nums) - 1):
            self.category_startidx.append(self.category_startidx[-1] +
                                          self.category_nums[i])
        assert self.loss_type in ['individual', 'all']
        assert self.reduction in ['mean', 'sum']

    def _forward(self, cls_score, label, mask, category_mask):
        

        if self.loss_type == 'all':
            loss_cls = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none')
            if self.with_mask:
                w_loss_cls = mask * loss_cls
                w_loss_cls = torch.sum(w_loss_cls, dim=1)
                if self.reduction == 'mean':
                    w_loss_cls = w_loss_cls / torch.sum(mask, dim=1)
                w_loss_cls = torch.mean(w_loss_cls)
                return dict(loss_cls=w_loss_cls)

            if self.reduction == 'sum':
                loss_cls = torch.sum(loss_cls, dim=-1)
            return dict(loss_cls=torch.mean(loss_cls))

        if self.loss_type == 'individual':
            losses = {}
            loss_weights = {}
            for name, num, start_idx in zip(self.categories,
                                            self.category_nums,
                                            self.category_startidx):
                category_score = cls_score[:, start_idx:start_idx + num]
                category_label = label[:, start_idx:start_idx + num]
                category_loss = F.binary_cross_entropy_with_logits(
                    category_score, category_label, reduction='none')
                if self.reduction == 'mean':
                    category_loss = torch.mean(category_loss, dim=1)
                elif self.reduction == 'sum':
                    category_loss = torch.sum(category_loss, dim=1)

                idx = self.categories.index(name)
                if self.with_mask:
                    category_mask_i = category_mask[:, idx].reshape(-1)


                    if torch.sum(category_mask_i) < 0.5:
                        losses[f'{name}_LOSS'] = torch.tensor(.0).cuda()
                        loss_weights[f'{name}_LOSS'] = .0
                        continue
                    category_loss = torch.sum(category_loss * category_mask_i)
                    category_loss = category_loss / torch.sum(category_mask_i)
                else:
                    category_loss = torch.mean(category_loss)



                losses[f'{name}_LOSS'] = category_loss
                loss_weights[f'{name}_LOSS'] = self.category_loss_weights[idx]
            loss_weight_sum = sum(loss_weights.values())
            loss_weights = {
                k: v / loss_weight_sum
                for k, v in loss_weights.items()
            }
            loss_cls = sum([losses[k] * loss_weights[k] for k in losses])
            losses['loss_cls'] = loss_cls

            losses.update({
                k + '_weight': torch.tensor(v).to(losses[k].device)
                for k, v in loss_weights.items()
            })

            return losses
        else:
            raise ValueError("loss_type should be 'all' or 'individual', "
                             f'but got {self.loss_type}')
