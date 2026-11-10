
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from .binary_logistic_regression_loss import binary_logistic_regression_loss


@MODELS.register_module()
class BMNLoss(nn.Module):
    

    @staticmethod
    def tem_loss(pred_start, pred_end, gt_start, gt_end):
        
        loss_start = binary_logistic_regression_loss(pred_start, gt_start)
        loss_end = binary_logistic_regression_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    @staticmethod
    def pem_reg_loss(pred_score,
                     gt_iou_map,
                     mask,
                     high_temporal_iou_threshold=0.7,
                     low_temporal_iou_threshold=0.3):
        
        u_hmask = (gt_iou_map > high_temporal_iou_threshold).float()
        u_mmask = ((gt_iou_map <= high_temporal_iou_threshold) &
                   (gt_iou_map > low_temporal_iou_threshold)).float()
        u_lmask = ((gt_iou_map <= low_temporal_iou_threshold) &
                   (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(
            loss * torch.ones_like(weights)) / torch.sum(weights)

        return loss

    @staticmethod
    def pem_cls_loss(pred_score,
                     gt_iou_map,
                     mask,
                     threshold=0.9,
                     ratio_range=(1.05, 21),
                     eps=1e-5):
        
        pmask = (gt_iou_map > threshold).float()
        nmask = (gt_iou_map <= threshold).float()
        nmask = nmask * mask

        num_positive = max(torch.sum(pmask), 1)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        ratio = torch.clamp(ratio, ratio_range[0], ratio_range[1])

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1 * torch.log(pred_score + eps) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + eps) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss

    def forward(self,
                pred_bm,
                pred_start,
                pred_end,
                gt_iou_map,
                gt_start,
                gt_end,
                bm_mask,
                weight_tem=1.0,
                weight_pem_reg=10.0,
                weight_pem_cls=1.0):
        
        pred_bm_reg = pred_bm[:, 0].contiguous()
        pred_bm_cls = pred_bm[:, 1].contiguous()
        gt_iou_map = gt_iou_map * bm_mask

        pem_reg_loss = self.pem_reg_loss(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end)
        loss = (
            weight_tem * tem_loss + weight_pem_reg * pem_reg_loss +
            weight_pem_cls * pem_cls_loss)
        return loss, tem_loss, pem_reg_loss, pem_cls_loss
