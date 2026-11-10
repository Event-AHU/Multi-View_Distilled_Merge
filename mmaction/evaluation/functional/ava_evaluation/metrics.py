















import numpy as np


def compute_precision_recall(scores, labels, num_gt):
    
    if (not isinstance(labels, np.ndarray) or labels.dtype != bool
            or len(labels.shape) != 1):
        raise ValueError('labels must be single dimension bool numpy array')

    if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError('scores must be single dimension numpy array')

    if num_gt < np.sum(labels):
        raise ValueError(
            'Number of true positives must be smaller than num_gt.')

    if len(scores) != len(labels):
        raise ValueError('scores and labels must be of the same size.')

    if num_gt == 0:
        return None, None

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    
    if precision is None:
        if recall is not None:
            raise ValueError('If precision is None, recall must also be None')
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
            recall, np.ndarray):
        raise ValueError('precision and recall must be numpy array')
    if precision.dtype != np.float64 or recall.dtype != np.float64:
        raise ValueError('input must be float numpy array.')
    if len(precision) != len(recall):
        raise ValueError('precision and recall must be of the same size.')
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError('Precision must be in the range of [0, 1].')
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError('recall must be in the range of [0, 1].')
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError('recall must be a non-decreasing array')

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])


    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
    

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            num_gt_imgs_per_class == 0, np.nan,
            num_images_correctly_detected_per_class / num_gt_imgs_per_class)
