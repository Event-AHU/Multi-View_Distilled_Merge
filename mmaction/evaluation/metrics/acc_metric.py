
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mean_class_accuracy,
                                 mmit_mean_average_precision, top_k_accuracy)
from mmaction.registry import METRICS
import pdb


def to_tensor(value):
    
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class AccMetric(BaseMetric):
    
    default_prefix: Optional[str] = 'acc'

    def __init__(
            self,
            metric_list: Optional[Union[str, Tuple[str]]] = ('top_k_accuracy', 'mean_class_accuracy'),
            collect_device: str = 'cpu',
            metric_options: Optional[Dict] = dict(
                top_k_accuracy=dict(topk=(1, 5))),
            prefix: Optional[str] = None,
            num_classes: Optional[int] = None):




        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, 'f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list


        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

            if metric in [
                    'mmit_mean_average_precision', 'mean_average_precision'
            ]:
                assert type(num_classes) == int

        self.metrics = metrics
        self.metric_options = metric_options
        self.num_classes = num_classes

    def process(self, data_batch, data_samples):
        import copy
        data_samples = copy.deepcopy(data_samples)

        for data_sample in data_samples:
            result = dict()


            if isinstance(data_sample, dict):
                pred = data_sample['pred_scores']
                gt = data_sample['gt_labels']
                label = gt['item'].item()
            else:

                pred = data_sample.pred_scores
                gt = data_sample.gt_labels if hasattr(data_sample, 'gt_labels') else data_sample.gt_label

                if isinstance(gt, dict):
                    label = gt['item'].item()
                else:

                    label = int(gt.item()) if hasattr(gt, 'item') else int(gt)


            pred_np = {}
            for item_name, score in pred.items():
                if hasattr(score, 'detach'):
                    pred_np[item_name] = score.detach().cpu().numpy()
                else:
                    pred_np[item_name] = score
            result['pred'] = pred_np
            result['label'] = label
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        

        labels = [x['label'] for x in results]

        if len(results[0]['pred']) == 1:
            preds = [x['pred']['item'] for x in results]
            return self.calculate(preds, labels)

        eval_results = dict()
        for item_name in results[0]['pred'].keys():
            preds = [x['pred'][item_name] for x in results]




            eval_result = self.calculate(preds, labels)
            eval_results.update(
                {f'{item_name}_{k}': v
                 for k, v in eval_result.items()})


        if len(results[0]['pred']) == 2 and\
                'rgb' in results[0]['pred'] and\
                'pose' in results[0]['pred']:

            rgb = [x['pred']['rgb'] for x in results]
            pose = [x['pred']['pose'] for x in results]

            preds = {
                '1:1': get_weighted_score([rgb, pose], [1, 1]),
                '2:1': get_weighted_score([rgb, pose], [2, 1]),
                '1:2': get_weighted_score([rgb, pose], [1, 2])
            }
            for k in preds:
                eval_result = self.calculate(preds[k], labels)
                eval_results.update({
                    f'RGBPose_{k}_{key}': v
                    for key, v in eval_result.items()
                })

        return eval_results

    def calculate(self, preds: List[np.ndarray], labels: List[int]) -> Dict:
        




        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, 'f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )


                top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, top_k_acc):

                    eval_results[f'top{k}'] = acc

            if metric == 'mean_class_accuracy':
                mean1 = mean_class_accuracy(preds, labels)
                eval_results['mean1'] = mean1

            if metric in [
                    'mean_average_precision',
                    'mmit_mean_average_precision',
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in labels
                ]

                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(preds, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP

                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(preds, gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP

        return eval_results

    @staticmethod
    def label2array(num, label):
        
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr


@METRICS.register_module()
class ConfusionMatrix(BaseMetric):
    
    default_prefix = 'confusion_matrix'

    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_scores = data_sample.get('pred_scores')
            gt_label = data_sample['gt_labels']['item']
            if pred_scores is not None:
                pred_label = pred_scores['item'].argmax(dim=0, keepdim=True)
                self.num_classes = pred_scores['item'].size(0)
            else:
                pred_label = data_sample['pred_labels']['item']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': gt_label
            })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)
        return {'result': confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        
        pred = to_tensor(pred)
        target_label = to_tensor(target).int()

        assert pred.size(0) == target_label.size(0),\
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target_label.size(0)}).'
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None,\
                'Please specify the `num_classes` if the `pred` is labels '\
                'intead of scores.'
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes**2)
            matrix = matrix.reshape(num_classes, num_classes)

        return matrix

    @staticmethod
    def plot(confusion_matrix: torch.Tensor,
             include_values: bool = False,
             cmap: str = 'viridis',
             classes: Optional[List[str]] = None,
             colorbar: bool = True,
             show: bool = True):
        
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)


            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i,
                                                     j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], '.2g')
                text_d = format(confusion_matrix[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha='center', va='center', color=color)

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel='True label',
            xlabel='Predicted label',
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))

        fig.autofmt_xdate(ha='center')

        if show:
            plt.show()
        return fig
