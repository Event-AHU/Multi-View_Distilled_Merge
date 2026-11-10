
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmaction.structures.bbox import bbox_target
from mmaction.utils import InstanceList

try:
    from mmdet.models.task_modules.samplers import SamplingResult
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    from mmaction.utils import SamplingResult
    mmdet_imported = False




from packaging import version as pv

if pv.parse(torch.__version__) < pv.parse('1.10'):

    def cross_entropy_loss(input, target, reduction='None'):
        input = input.log_softmax(dim=-1)
        loss = -(input * target).sum(dim=-1)
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
else:
    cross_entropy_loss = F.cross_entropy


class BBoxHeadAVA(nn.Module):
    

    def __init__(
            self,
            temporal_pool_type: str = 'avg',
            spatial_pool_type: str = 'max',
            in_channels: int = 2048,
            focal_gamma: float = 0.,
            focal_alpha: float = 1.,
            num_classes: int = 81,
            dropout_ratio: float = 0,
            dropout_before_pool: bool = True,
            topk: Union[int, Tuple[int]] = (3, 5),
            multilabel: bool = True,
            mlp_head: bool = False) -> None:
        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')


        assert all([k < num_classes for k in self.topk])

        in_channels = self.in_channels

        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        if mlp_head:
            self.fc_cls = nn.Sequential(
                nn.Linear(in_channels, in_channels), nn.ReLU(),
                nn.Linear(in_channels, num_classes))
        else:
            self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self) -> None:
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.temporal_pool(x)
        x = self.spatial_pool(x)

        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return cls_score

    @staticmethod
    def get_targets(sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict) -> tuple:
        pos_proposals = [res.pos_priors for res in sampling_results]
        neg_proposals = [res.neg_priors for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_labels,
                                  rcnn_train_cfg)
        return cls_targets

    @staticmethod
    def get_recall_prec(pred_vec: Tensor, target_vec: Tensor) -> tuple:
        
        correct = pred_vec & target_vec
        recall = correct.sum(1) / target_vec.sum(1).float()
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    @staticmethod
    def topk_to_matrix(probs: Tensor, k: int) -> Tensor:
        
        topk_labels = probs.topk(k, 1, True, True)[1]
        topk_matrix = probs.new_full(probs.size(), 0, dtype=torch.bool)
        for i in range(probs.shape[0]):
            topk_matrix[i, topk_labels[i]] = 1
        return topk_matrix

    def topk_accuracy(self,
                      pred: Tensor,
                      target: Tensor,
                      thr: float = 0.5) -> tuple:
        

        target_bool = target > 0.5


        if self.multilabel:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)


        if self.multilabel:
            pred_bool = pred > thr
        else:
            pred_bool = self.topk_to_matrix(pred, 1)
        recall_thr, prec_thr = self.get_recall_prec(pred_bool, target_bool)


        recalls_k, precs_k = [], []
        for k in self.topk:
            pred_bool = self.topk_to_matrix(pred, k)
            recall, prec = self.get_recall_prec(pred_bool, target_bool)
            recalls_k.append(recall)
            precs_k.append(prec)


        return recall_thr, prec_thr, recalls_k, precs_k

    def loss_and_target(self, cls_score: Tensor, rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict, **kwargs) -> dict:
        
        cls_targets = self.get_targets(sampling_results, rcnn_train_cfg)
        labels, _ = cls_targets

        losses = dict()

        if cls_score is not None:
            labels = labels[:, 1:]
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds, 1:]
            labels = labels[pos_inds]



            recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]



            if not self.multilabel:
                labels = labels / labels.sum(dim=1, keepdim=True)



            if self.multilabel:
                loss_func = F.binary_cross_entropy_with_logits
            else:
                loss_func = cross_entropy_loss


            loss = loss_func(cls_score, labels, reduction='none')
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            losses['loss_action_cls'] = torch.mean(F_loss)

        return dict(loss_bbox=losses, bbox_targets=cls_targets)

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        **kwargs) -> InstanceList:
        
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                img_meta=img_meta,
                rcnn_test_cfg=rcnn_test_cfg,
                **kwargs)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(self,
                                roi: Tensor,
                                cls_score: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: Optional[ConfigDict] = None,
                                **kwargs) -> InstanceData:
        
        results = InstanceData()


        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))


        if cls_score is not None:
            if self.multilabel:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = roi[:, 1:]
        assert bboxes.shape[-1] == 4


        img_h, img_w = img_meta['img_shape']
        if img_meta.get('flip', False):
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_


        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        crop_quadruple = img_meta.get('crop_quadruple', np.array([0, 0, 1, 1]))
        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)

        results.bboxes = bboxes
        results.scores = scores

        return results


if mmdet_imported:
    MMDET_MODELS.register_module()(BBoxHeadAVA)
