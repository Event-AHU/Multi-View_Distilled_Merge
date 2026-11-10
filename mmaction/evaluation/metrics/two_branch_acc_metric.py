import torch
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS


def _topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, topk=(1,)):
    
    maxk = max(topk)
    _, pred = scores.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[k] = (correct_k / labels.numel()).item()
    return res


@METRICS.register_module()
class TwoBranchAccMetric(BaseMetric):
    
    def __init__(self,
                 other_branch='sst',
                 topk=(1,),
                 prefix_main='tsc',
                 prefix_other='sst',
                 collect_device='cpu'):
        super().__init__(collect_device=collect_device)
        self.other_branch = other_branch
        self.topk = tuple(topk)
        self.prefix_main = prefix_main
        self.prefix_other = prefix_other

    def process(self, data_batch, data_samples):
        for ds in data_samples:

            if hasattr(ds, 'gt_label'):
                lab = ds.gt_label
            elif hasattr(ds, 'gt_labels'):
                lab = ds.gt_labels
            else:
                raise AttributeError('Missing gt_label/gt_labels in data_sample.')

            if torch.is_tensor(lab):
                lab = lab.squeeze().detach().cpu().long()
                lab = int(lab.item())
            else:
                lab = int(lab)


            s_main = ds.pred_score
            attr = f'pred_score_{self.other_branch}'
            if not hasattr(ds, attr):
                raise AttributeError(f'Missing `{attr}` in data_sample. '
                                    f'Please set it in wrapper.predict().')
            s_other = getattr(ds, attr)

            self.results.append({
                'label': lab,
                'main': s_main.detach().cpu(),
                'other': s_other.detach().cpu()
            })

    def compute_metrics(self, results):
        labels = torch.tensor([r['label'] for r in results], dtype=torch.long)
        main_scores = torch.stack([r['main'] for r in results], dim=0)
        other_scores = torch.stack([r['other'] for r in results], dim=0)

        m = _topk_accuracy(main_scores, labels, self.topk)
        o = _topk_accuracy(other_scores, labels, self.topk)

        out = {}
        for k in self.topk:
            out[f'{self.prefix_main}/acc/top{k}'] = m[k]
            out[f'{self.prefix_other}/acc/top{k}'] = o[k]
        return out
