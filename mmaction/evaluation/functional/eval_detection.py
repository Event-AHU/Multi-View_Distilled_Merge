
import json

import numpy as np
from mmengine.logging import MMLogger, print_log

from .accuracy import interpolated_precision_recall, pairwise_temporal_iou


class ActivityNetLocalization:
    

    def __init__(self,
                 ground_truth_filename=None,
                 prediction_filename=None,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.ground_truth_filename = ground_truth_filename
        self.prediction_filename = prediction_filename
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.logger = MMLogger.get_current_instance()

        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            log_msg = (
                '[INIT] Loaded ground_truth from '
                f'{self.ground_truth_filename}, prediction from '
                f'{self.prediction_filename}.\n'
                f'Number of ground truth instances: {len(self.ground_truth)}\n'
                f'Number of predictions: {len(self.prediction)}\n'
                f'Fixed threshold for tiou score: {self.tiou_thresholds}')
            print_log(log_msg, logger=self.logger)

    @staticmethod
    def _import_ground_truth(ground_truth_filename):
        
        with open(ground_truth_filename, 'r') as f:
            data = json.load(f)

        activity_index, class_idx = {}, 0
        ground_truth = []
        for video_id, video_info in data.items():
            for anno in video_info['annotations']:
                if anno['label'] not in activity_index:
                    activity_index[anno['label']] = class_idx
                    class_idx += 1

                ground_truth_item = {}
                ground_truth_item['video-id'] = video_id[2:]
                ground_truth_item['t-start'] = float(anno['segment'][0])
                ground_truth_item['t-end'] = float(anno['segment'][1])
                ground_truth_item['label'] = activity_index[anno['label']]
                ground_truth.append(ground_truth_item)

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        
        with open(prediction_filename, 'r') as f:
            data = json.load(f)

        prediction = []
        for video_id, video_info in data['results'].items():
            for result in video_info:
                prediction_item = dict()
                prediction_item['video-id'] = video_id
                prediction_item['label'] = self.activity_index[result['label']]
                prediction_item['t-start'] = float(result['segment'][0])
                prediction_item['t-end'] = float(result['segment'][1])
                prediction_item['score'] = result['score']
                prediction.append(prediction_item)

        return prediction

    def wrapper_compute_average_precision(self):
        
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))


        ground_truth_by_label = []
        prediction_by_label = []
        for i in range(len(self.activity_index)):
            ground_truth_by_label.append([])
            prediction_by_label.append([])
        for gt in self.ground_truth:
            ground_truth_by_label[gt['label']].append(gt)
        for pred in self.prediction:
            prediction_by_label[pred['label']].append(pred)

        for i in range(len(self.activity_index)):
            ap_result = compute_average_precision_detection(
                ground_truth_by_label[i], prediction_by_label[i],
                self.tiou_thresholds)
            ap[:, i] = ap_result

        return ap

    def evaluate(self):
        
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        return self.mAP, self.average_mAP


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1

    prediction.sort(key=lambda x: -x['score'])

    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))


    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)


    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        tiou_arr = pairwise_temporal_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
        tiou_arr = tiou_arr.reshape(-1)

        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue

                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap
