
import torch
import torch.nn as nn

from mmaction.registry import MODELS


def binary_logistic_regression_loss(reg_score,
                                    label,
                                    threshold=0.5,
                                    ratio_range=(1.05, 21),
                                    eps=1e-5):
    
    label = label.view(-1).to(reg_score.device)
    reg_score = reg_score.contiguous().view(-1)

    pmask = (label > threshold).float().to(reg_score.device)
    num_positive = max(torch.sum(pmask), 1)
    num_entries = len(label)
    ratio = num_entries / num_positive

    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])

    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * torch.log(reg_score + eps) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


@MODELS.register_module()
class BinaryLogisticRegressionLoss(nn.Module):
    

    def forward(self,
                reg_score,
                label,
                threshold=0.5,
                ratio_range=(1.05, 21),
                eps=1e-5):
        

        return binary_logistic_regression_loss(reg_score, label, threshold,
                                               ratio_range, eps)
