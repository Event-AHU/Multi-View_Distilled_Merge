
import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_iop(proposal_min, proposal_max, gt_min, gt_max):
    
    len_anchors = np.array(proposal_max - proposal_min)
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def soft_nms(proposals, alpha, low_threshold, high_threshold, top_k):
    
    proposals = proposals[proposals[:, -1].argsort()[::-1]]
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, -1])
    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 0 and len(rscore) <= top_k:
        max_index = np.argmax(tscore)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = temporal_iou(tstart[max_index], tend[max_index],
                                np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)

        for idx, _ in enumerate(tscore):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (high_threshold -
                                                  low_threshold) * max_width:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    rstart = np.array(rstart).reshape(-1, 1)
    rend = np.array(rend).reshape(-1, 1)
    rscore = np.array(rscore).reshape(-1, 1)
    new_proposals = np.concatenate((rstart, rend, rscore), axis=1)
    return new_proposals


def post_processing(result, video_info, soft_nms_alpha, soft_nms_low_threshold,
                    soft_nms_high_threshold, post_process_top_k,
                    feature_extraction_interval):
    
    if len(result) > 1:
        result = soft_nms(result, soft_nms_alpha, soft_nms_low_threshold,
                          soft_nms_high_threshold, post_process_top_k)

    result = result[result[:, -1].argsort()[::-1]]
    video_duration = float(
        video_info['duration_frame'] // feature_extraction_interval *
        feature_extraction_interval
    ) / video_info['duration_frame'] * video_info['duration_second']
    proposal_list = []

    for j in range(min(post_process_top_k, len(result))):
        proposal = {}
        proposal['score'] = float(result[j, -1])
        proposal['segment'] = [
            max(0, result[j, 0]) * video_duration,
            min(1, result[j, 1]) * video_duration
        ]
        proposal_list.append(proposal)
    return proposal_list
