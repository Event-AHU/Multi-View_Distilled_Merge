
import torch
from torch import Tensor

try:
    from mmdet.models.task_modules import AssignResult, MaxIoUAssigner
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_TASK_UTILS.register_module()
    class MaxIoUAssignerAVA(MaxIoUAssigner):
        



        def assign_wrt_overlaps(self, overlaps: Tensor,
                                gt_labels: Tensor) -> AssignResult:
            
            num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)


            assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)

            if num_gts == 0 or num_bboxes == 0:

                max_overlaps = overlaps.new_zeros((num_bboxes, ))
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
                if num_gts == 0:

                    assigned_gt_inds[:] = 0
                return AssignResult(
                    num_gts=num_gts,
                    gt_inds=assigned_gt_inds,
                    max_overlaps=max_overlaps,
                    labels=assigned_labels)



            max_overlaps, argmax_overlaps = overlaps.max(dim=0)


            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)



            if isinstance(self.neg_iou_thr, float):
                assigned_gt_inds[(max_overlaps >= 0)
                                 & (max_overlaps < self.neg_iou_thr)] = 0
            elif isinstance(self.neg_iou_thr, tuple):
                assert len(self.neg_iou_thr) == 2
                assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                                 & (max_overlaps < self.neg_iou_thr[1])] = 0


            pos_inds = max_overlaps >= self.pos_iou_thr
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

            if self.match_low_quality:








                for i in range(num_gts):
                    if gt_max_overlaps[i] >= self.min_pos_iou:
                        if self.gt_max_assign_all:
                            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                            assigned_gt_inds[max_iou_inds] = i + 1
                        else:
                            assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1


            assert len(gt_labels[0]) > 1
            assigned_labels = assigned_gt_inds.new_zeros(
                (num_bboxes, len(gt_labels[0])), dtype=torch.float32)


            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

else:

    class MaxIoUAssignerAVA:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Failed to import `AssignResult`, `MaxIoUAssigner` from '
                '`mmdet.core.bbox` or failed to import `TASK_UTILS` from '
                '`mmdet.registry`. The class `MaxIoUAssignerAVA` is '
                'invalid. ')
