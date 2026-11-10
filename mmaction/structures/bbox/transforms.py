
import numpy as np
import torch


def bbox2result(bboxes: torch.Tensor,
                labels: torch.Tensor,
                num_classes: int,
                thr: float = 0.01) -> list:
    
    if bboxes.shape[0] == 0:
        return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))

    bboxes = bboxes.cpu().numpy()
    scores = labels.cpu().numpy()


    assert scores.shape[-1] > 1


    if not hasattr(thr, '__len__'):
        multilabel = thr >= 0
        thr = (thr, ) * num_classes
    else:
        multilabel = True


    assert scores.shape[1] == num_classes
    assert len(thr) == num_classes

    result = []
    for i in range(num_classes - 1):
        if multilabel:
            where = (scores[:, i + 1] > thr[i + 1])
        else:
            where = (scores[:, 1:].argmax(axis=1) == i)
        result.append(
            np.concatenate((bboxes[where, :4], scores[where, i + 1:i + 2]),
                           axis=1))
    return result
