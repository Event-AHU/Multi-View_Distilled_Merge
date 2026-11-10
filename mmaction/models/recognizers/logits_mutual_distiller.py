import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModel
from mmaction.registry import MODELS


@MODELS.register_module()
class LogitsMutualDistiller(BaseModel):
    def __init__(
        self,
        branches: dict,
        io_map: dict,
        kd_temperature: float = 4.0,
        kd_weight: float = 0.5,
        ce_weight: float = 0.5,
        pred_branch: str = 'tsc',
        branch_devices: dict = None,
        data_preprocessor=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        self.branches = nn.ModuleDict({k: MODELS.build(v) for k, v in branches.items()})
        self.io_map = io_map
        self.T = float(kd_temperature)
        self.kd_w = float(kd_weight)
        self.ce_w = float(ce_weight)
        self.pred_branch = pred_branch
        self.branch_devices = branch_devices or {}

        self._place_branches()
        self._loss_device = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _place_branches(self):
        for name, dev in self.branch_devices.items():
            if name in self.branches and dev is not None:
                self.branches[name].to(torch.device(dev))

    def to(self, *args, **kwargs):

        super().to(*args, **kwargs)
        self._place_branches()
        return self

    def _branch_device(self, name: str) -> torch.device:
        if name in self.branch_devices:
            return torch.device(self.branch_devices[name])
        return next(self.branches[name].parameters()).device

    def _move_data_samples(self, data_samples, device):
        if data_samples is None:
            return None

        return [ds.to(device) for ds in data_samples]

    def _branch_logits(self, name, inputs_dict, data_samples):
        m = self.branches[name]
        dev = self._branch_device(name)

        in_key = self.io_map[name]['input']
        ev_key = self.io_map[name].get('event', None)

        x = inputs_dict[in_key].to(dev, non_blocking=True)
        ev = inputs_dict[ev_key].to(dev, non_blocking=True) if ev_key is not None else None
        ds = self._move_data_samples(data_samples, dev)

        logits = m.forward(x, ev, data_samples=ds, mode='tensor', stage='head')
        return logits, ds, dev

    def loss(self, inputs, data_samples):

        logits = {}
        ds_on_dev = {}
        devs = {}
        for k in self.branches.keys():
            lg, ds_k, dev_k = self._branch_logits(k, inputs, data_samples)
            logits[k] = lg
            ds_on_dev[k] = ds_k
            devs[k] = dev_k


        losses = {}
        for k, m in self.branches.items():
            ce_dict = m.cls_head.loss_by_feat(logits[k], ds_on_dev[k])
            for kk, vv in ce_dict.items():
                losses[f'{k}.{kk}'] = vv * self.ce_w if 'loss' in kk else vv


        names = list(self.branches.keys())
        if self.kd_w > 0 and len(names) >= 2:
            for n in names:
                student = logits[n]
                dev = devs[n]

                teacher_sum = None
                cnt = 0
                for other in names:
                    if other == n:
                        continue
                    t = logits[other].detach().to(dev, non_blocking=True)
                    teacher_sum = t if teacher_sum is None else (teacher_sum + t)
                    cnt += 1
                teacher = teacher_sum / max(cnt, 1)

                kd = F.kl_div(
                    F.log_softmax(student / self.T, dim=1),
                    F.softmax(teacher / self.T, dim=1),
                    reduction='batchmean'
                ) * (self.T * self.T)

                losses[f'{n}.loss_kd'] = kd * self.kd_w

        master = self._loss_device.device
        for k, v in list(losses.items()):
            if torch.is_tensor(v) and v.device != master:
                losses[k] = v.to(master, non_blocking=True)
        return losses

    def predict(self, inputs, data_samples):
        
        assert data_samples is not None


        logits = {}
        for name in self.branches.keys():
            lg, _, _ = self._branch_logits(name, inputs, data_samples)
            logits[name] = lg


        scores_cpu = {
            name: F.softmax(lg, dim=1).detach().cpu()
            for name, lg in logits.items()
        }


        outputs = []
        for i, ds in enumerate(data_samples):

            if isinstance(ds, dict):
                gt_item = ds['gt_labels']['item']
            else:

                if hasattr(ds, 'gt_labels') and hasattr(ds.gt_labels, 'item'):
                    gt_item = ds.gt_labels.item
                elif hasattr(ds, 'gt_label') and hasattr(ds.gt_label, 'item'):
                    gt_item = ds.gt_label.item
                else:
                    raise AttributeError('Cannot find gt label in data_sample.')

            if torch.is_tensor(gt_item):
                gt_item = gt_item.detach().cpu()
            else:
                gt_item = torch.tensor(gt_item)

            outputs.append(dict(
                pred_scores={name: scores_cpu[name][i] for name in scores_cpu.keys()},
                gt_labels=dict(item=gt_item)
            ))

        return outputs

    def forward(self, inputs, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            out = {}
            for k in self.branches.keys():
                lg, _, _ = self._branch_logits(k, inputs, data_samples)
                out[k] = lg
            return out
