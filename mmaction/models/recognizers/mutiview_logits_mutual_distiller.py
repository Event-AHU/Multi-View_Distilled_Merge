import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModel

from mmaction.registry import MODELS


class _FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend([
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PeerWeightNet(nn.Module):
    

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2 + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, student_feat: torch.Tensor, peer_feat: torch.Tensor,
                peer_conf: torch.Tensor, pair_sim: torch.Tensor) -> torch.Tensor:
        x = torch.cat([student_feat, peer_feat, peer_conf, pair_sim], dim=1)
        return self.net(x)


@MODELS.register_module()
class MutiviewLogitsMutualDistiller(BaseModel):
    

    def __init__(
        self,
        branches: dict,
        io_map: dict,
        kd_temperature: float = 4.0,
        kd_weight: float = 0.5,
        ce_weight: float = 0.5,
        kd_start_epoch: int = 5,
        pred_branch: str = 'mv_frame',
        branch_devices: dict = None,
        feature_kd_weight: float = 0.0,
        feature_kd_start_epoch: int = None,
        feature_loss_type: str = 'cosine',
        feature_stage: str = 'backbone',
        feature_projector_hidden: int = 1024,
        feature_projector_out: int = 512,
        feature_projector_dropout: float = 0.0,
        peer_weight_hidden: int = 256,
        feature_relation_weight: float = 0.0,
        class_prior=None,
        class_prior_weight: float = 0.0,
        enable_logit_fusion: bool = False,
        logit_fusion_weights=None,
        logit_fusion_temperatures=None,
        logit_fusion_name: str = 'fused',
        data_preprocessor=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        self.branches = nn.ModuleDict({k: MODELS.build(v) for k, v in branches.items()})
        self.branch_names = list(self.branches.keys())
        self.io_map = io_map
        self.T = float(kd_temperature)
        self.kd_w = float(kd_weight)
        self.ce_w = float(ce_weight)
        self.kd_start_epoch = int(kd_start_epoch)
        self.pred_branch = pred_branch
        self.branch_devices = branch_devices or {}


        self.enable_logit_fusion = bool(enable_logit_fusion)
        self.logit_fusion_name = str(logit_fusion_name)
        self.logit_fusion_weights = self._normalize_branch_float_dict(
            logit_fusion_weights,
            default=1.0 / max(len(self.branch_names), 1),
            normalize=True)
        self.logit_fusion_temperatures = self._normalize_branch_float_dict(
            logit_fusion_temperatures,
            default=1.0,
            normalize=False)

        self.feature_kd_weight = float(feature_kd_weight)
        self.feature_kd_start_epoch = self.kd_start_epoch if feature_kd_start_epoch is None else int(feature_kd_start_epoch)
        self.feature_loss_type = str(feature_loss_type).lower()
        self.feature_stage = str(feature_stage)
        self.feature_relation_weight = float(feature_relation_weight)
        self.class_prior_weight = float(class_prior_weight)

        if self.feature_stage not in ('backbone', 'neck'):
            raise ValueError(
                f'feature_stage must be "backbone" or "neck", got {self.feature_stage!r}.')
        if self.feature_loss_type not in ('mse', 'smooth_l1', 'cosine'):
            raise ValueError(
                "feature_loss_type must be one of {'mse', 'smooth_l1', 'cosine'}, "
                f'but got {feature_loss_type!r}.')

        self.feature_projectors = nn.ModuleDict()
        for name in self.branch_names:
            in_dim = getattr(self.branches[name].cls_head, 'in_channels', None)
            if in_dim is None:
                raise AttributeError(
                    f'Branch {name!r} cls_head must expose in_channels for projector init.')
            self.feature_projectors[name] = _FeatureProjector(
                in_dim=int(in_dim),
                hidden_dim=int(feature_projector_hidden),
                out_dim=int(feature_projector_out),
                dropout=float(feature_projector_dropout),
            )

        self.peer_weight_nets = nn.ModuleDict({
            self._pair_key(student, peer): _PeerWeightNet(
                feat_dim=int(feature_projector_out),
                hidden_dim=int(peer_weight_hidden),
            )
            for student in self.branch_names
            for peer in self.branch_names
            if peer != student
        })

        prior_tensor = self._build_class_prior_tensor(class_prior)
        if prior_tensor is None:
            self.register_buffer('class_prior', torch.empty(0), persistent=False)
            self._has_class_prior = False
        else:
            self.register_buffer('class_prior', prior_tensor.float(), persistent=False)
            self._has_class_prior = True

        self._place_branches()
        self._loss_device = nn.Parameter(torch.zeros(1), requires_grad=False)

    @staticmethod
    def _pair_key(student: str, peer: str) -> str:
        return f'{student}__{peer}'

    def _normalize_branch_float_dict(self, values, default: float, normalize: bool):
        if values is None:
            out = {name: float(default) for name in self.branch_names}
        elif isinstance(values, dict):
            out = {name: float(values.get(name, default)) for name in self.branch_names}
        elif isinstance(values, (list, tuple)):
            if len(values) != len(self.branch_names):
                raise ValueError(
                    f'Expected {len(self.branch_names)} values, got {len(values)}.')
            out = {name: float(values[idx]) for idx, name in enumerate(self.branch_names)}
        else:
            raise TypeError(
                'Fusion weights/temperatures must be None, dict, list, or tuple, '
                f'but got {type(values)!r}.')

        if normalize:
            total = sum(out.values())
            if abs(total) < 1e-12:
                raise ValueError(f'Fusion weight sum is zero: {out}')
            out = {k: v / total for k, v in out.items()}

        return out

    def _build_class_prior_tensor(self, class_prior):
        if class_prior is None:
            return None

        num_branches = len(self.branch_names)
        branch_to_idx = {name: idx for idx, name in enumerate(self.branch_names)}

        if torch.is_tensor(class_prior):
            tensor = class_prior.detach().float().cpu()
        elif isinstance(class_prior, (list, tuple)):
            tensor = torch.tensor(class_prior, dtype=torch.float32)
        elif isinstance(class_prior, dict):



            if all(name in class_prior for name in self.branch_names):
                cols = [torch.tensor(class_prior[name], dtype=torch.float32).view(-1, 1)
                        for name in self.branch_names]
                tensor = torch.cat(cols, dim=1)
            else:
                max_cls = max(int(k) for k in class_prior.keys()) + 1
                tensor = torch.zeros(max_cls, num_branches, dtype=torch.float32)
                for cls_idx, per_branch in class_prior.items():
                    if not isinstance(per_branch, dict):
                        raise TypeError('class_prior[class_idx] must be a dict of branch weights.')
                    for branch_name, value in per_branch.items():
                        tensor[int(cls_idx), branch_to_idx[branch_name]] = float(value)
        else:
            raise TypeError(
                'class_prior must be None, Tensor, list/tuple, or dict, '
                f'but got {type(class_prior)!r}.')

        if tensor.dim() != 2 or tensor.size(1) != num_branches:
            raise ValueError(
                f'class_prior must have shape [num_classes, {num_branches}], '
                f'but got {tuple(tensor.shape)}.')

        row_sum = tensor.sum(dim=1, keepdim=True)
        valid = row_sum.squeeze(1) > 0
        if valid.any():
            tensor[valid] = tensor[valid] / row_sum[valid]
        return tensor

    def _place_branches(self):
        for name, dev in self.branch_devices.items():
            if name in self.branches and dev is not None:
                device = torch.device(dev)
                self.branches[name].to(device)
                self.feature_projectors[name].to(device)
                for peer in self.branch_names:
                    if peer == name:
                        continue
                    self.peer_weight_nets[self._pair_key(name, peer)].to(device)

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

    def _extract_label_tensor(self, data_samples):
        if data_samples is None:
            return None
        labels = []
        for ds in data_samples:
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
                labels.append(gt_item.detach().view(-1).long().cpu())
            else:
                labels.append(torch.tensor([gt_item], dtype=torch.long))
        return torch.cat(labels, dim=0)

    def _prepare_feature(self, feat) -> torch.Tensor:
        if isinstance(feat, (list, tuple)):
            if len(feat) == 1:
                feat = feat[0]
            else:
                feat = torch.cat([self._prepare_feature(f) for f in feat], dim=1)
                return feat
        if not torch.is_tensor(feat):
            raise TypeError(f'Feature must be a Tensor, got {type(feat)}')
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return feat

    def _reduce_feat_for_kd(self, name: str, feat: torch.Tensor) -> torch.Tensor:
        
        num_segments = int(getattr(self.branches[name].cls_head, 'num_segments', 1))
        if feat.dim() != 2:
            feat = feat.flatten(1)
        if num_segments > 1 and feat.size(0) % num_segments == 0:
            feat = feat.view(-1, num_segments, feat.size(1)).mean(dim=1)
        return feat

    def _compute_feature_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.feature_loss_type == 'mse':
            return F.mse_loss(student, teacher)
        if self.feature_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(student, teacher)
        if self.feature_loss_type == 'cosine':
            student = F.normalize(student, p=2, dim=1)
            teacher = F.normalize(teacher, p=2, dim=1)
            return 1.0 - F.cosine_similarity(student, teacher, dim=1).mean()
        raise RuntimeError(f'Unsupported feature_loss_type={self.feature_loss_type}')

    def _compute_relation_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.size(0) <= 1:
            return student.new_zeros(())
        student = F.normalize(student, p=2, dim=1)
        teacher = F.normalize(teacher, p=2, dim=1)
        student_rel = student @ student.transpose(0, 1)
        teacher_rel = teacher @ teacher.transpose(0, 1)
        return F.mse_loss(student_rel, teacher_rel)

    def _reduce_logits_for_weight(self,
                                  name: str,
                                  logit: torch.Tensor,
                                  target_batch: int = None) -> torch.Tensor:
        
        num_segments = int(getattr(self.branches[name].cls_head, 'num_segments', 1))
        if logit.dim() != 2:
            logit = logit.flatten(1)
        if target_batch is not None and logit.size(0) == target_batch:
            return logit
        if num_segments > 1 and logit.size(0) % num_segments == 0:
            logit = logit.view(-1, num_segments, logit.size(1)).mean(dim=1)
        if target_batch is not None and logit.size(0) != target_batch:
            raise RuntimeError(
                f'Adaptive feature KD batch mismatch in {name}: '
                f'got logits batch={logit.size(0)}, expected {target_batch}.')
        return logit

    def _lookup_prior(self, label_cpu: torch.Tensor, peer_name: str, device: torch.device) -> torch.Tensor:
        if (not self._has_class_prior) or self.class_prior_weight <= 0:
            return torch.zeros(label_cpu.size(0), 1, device=device)
        if self.class_prior.numel() == 0:
            return torch.zeros(label_cpu.size(0), 1, device=device)
        branch_idx = self.branch_names.index(peer_name)
        safe_label = label_cpu.clamp(min=0, max=self.class_prior.size(0) - 1)
        prior = self.class_prior[safe_label, branch_idx].to(device=device, dtype=torch.float32)
        return prior.view(-1, 1)

    def _branch_forward(self, name, inputs_dict, data_samples):
        m = self.branches[name]
        dev = self._branch_device(name)

        in_key = self.io_map[name]['input']
        ev_key = self.io_map[name].get('event', None)

        x = inputs_dict[in_key].to(dev, non_blocking=True)
        ev = inputs_dict[ev_key].to(dev, non_blocking=True) if ev_key is not None else None
        ds = self._move_data_samples(data_samples, dev)

        feat_raw, head_kwargs = m.extract_feat(
            inputs=x,
            imgs_event=ev,
            stage=self.feature_stage,
            data_samples=ds)
        feat_head = self._prepare_feature(feat_raw)
        logits = m.cls_head(feat_head, **head_kwargs)
        feat_kd = self._reduce_feat_for_kd(name, feat_head)
        return logits, feat_kd, ds, dev

    def _get_current_epoch(self) -> int:
        epoch = 0
        try:
            from mmengine.logging import MessageHub
            message_hub = MessageHub.get_current_instance()

            if hasattr(message_hub, 'get_info'):
                val = message_hub.get_info('epoch')
                if val is not None:
                    if torch.is_tensor(val):
                        val = val.item()
                    return int(val)

            runtime_info = getattr(message_hub, 'runtime_info', None)
            if runtime_info is not None and 'epoch' in runtime_info:
                val = runtime_info['epoch']
                if torch.is_tensor(val):
                    val = val.item()
                return int(val)
        except Exception:
            pass
        return epoch

    def _kd_is_active(self) -> bool:
        current_epoch = self._get_current_epoch()
        return current_epoch >= self.kd_start_epoch

    def _feature_kd_is_active(self) -> bool:
        current_epoch = self._get_current_epoch()
        return current_epoch >= self.feature_kd_start_epoch

    def _build_adaptive_teacher(self,
                                student_name: str,
                                proj_feats: dict,
                                logits: dict,
                                label_cpu: torch.Tensor):
        student_dev = proj_feats[student_name].device
        student_feat = proj_feats[student_name]
        student_det = student_feat.detach()

        peer_scores = []
        peer_feats = []
        peer_names = []
        for peer_name in self.branch_names:
            if peer_name == student_name:
                continue
            peer_feat = proj_feats[peer_name].detach().to(student_dev, non_blocking=True)
            peer_logit = self._reduce_logits_for_weight(
                peer_name,
                logits[peer_name].detach().to(student_dev, non_blocking=True),
                target_batch=student_feat.size(0))
            peer_conf = F.softmax(peer_logit, dim=1).max(dim=1, keepdim=True).values
            pair_sim = F.cosine_similarity(student_det, peer_feat, dim=1).view(-1, 1)

            scorer = self.peer_weight_nets[self._pair_key(student_name, peer_name)]
            score = scorer(student_det, peer_feat, peer_conf, pair_sim)
            if label_cpu is not None and self.class_prior_weight > 0:
                score = score + self.class_prior_weight * self._lookup_prior(
                    label_cpu, peer_name, student_dev)

            peer_scores.append(score)
            peer_feats.append(peer_feat)
            peer_names.append(peer_name)

        score_tensor = torch.cat(peer_scores, dim=1)
        alpha = F.softmax(score_tensor, dim=1)

        teacher = torch.zeros_like(student_feat)
        for idx, peer_feat in enumerate(peer_feats):
            teacher = teacher + alpha[:, idx:idx + 1] * peer_feat
        return teacher, alpha, peer_names

    def loss(self, inputs, data_samples):
        logits = {}
        feats = {}
        proj_feats = {}
        ds_on_dev = {}
        devs = {}
        label_cpu = self._extract_label_tensor(data_samples)

        for name in self.branch_names:
            lg, ft, ds_k, dev_k = self._branch_forward(name, inputs, data_samples)
            logits[name] = lg
            feats[name] = ft
            proj_feats[name] = self.feature_projectors[name](ft)
            ds_on_dev[name] = ds_k
            devs[name] = dev_k

        losses = {}
        for name, model in self.branches.items():
            ce_dict = model.cls_head.loss_by_feat(logits[name], ds_on_dev[name])
            for loss_name, value in ce_dict.items():
                losses[f'{name}.{loss_name}'] = value * self.ce_w if 'loss' in loss_name else value

        if self._kd_is_active() and len(self.branch_names) >= 2:
            if self.kd_w > 0:
                for name in self.branch_names:
                    student = logits[name]
                    dev = devs[name]

                    teacher_sum = None
                    cnt = 0
                    for other in self.branch_names:
                        if other == name:
                            continue
                        t = logits[other].detach().to(dev, non_blocking=True)
                        teacher_sum = t if teacher_sum is None else (teacher_sum + t)
                        cnt += 1
                    teacher = teacher_sum / max(cnt, 1)

                    kd = F.kl_div(
                        F.log_softmax(student / self.T, dim=1),
                        F.softmax(teacher / self.T, dim=1),
                        reduction='batchmean') * (self.T * self.T)
                    losses[f'{name}.loss_kd'] = kd * self.kd_w

            if self.feature_kd_weight > 0 and self._feature_kd_is_active():
                for name in self.branch_names:
                    teacher_feat, alpha, peer_names = self._build_adaptive_teacher(
                        student_name=name,
                        proj_feats=proj_feats,
                        logits=logits,
                        label_cpu=label_cpu)
                    student_feat = proj_feats[name]

                    feat_loss = self._compute_feature_loss(student_feat, teacher_feat)
                    losses[f'{name}.loss_feat_kd'] = feat_loss * self.feature_kd_weight

                    if self.feature_relation_weight > 0:
                        rel_loss = self._compute_relation_loss(student_feat, teacher_feat)
                        losses[f'{name}.loss_feat_rel'] = rel_loss * self.feature_relation_weight

                    for idx, peer_name in enumerate(peer_names):
                        losses[f'{name}.peer_weight_{peer_name}'] = alpha[:, idx].mean().detach()

        master = self._loss_device.device
        for key, value in list(losses.items()):
            if torch.is_tensor(value) and value.device != master:
                losses[key] = value.to(master, non_blocking=True)
        return losses

    def predict(self, inputs, data_samples):
        assert data_samples is not None

        logits = {}
        for name in self.branch_names:
            lg, _, _, _ = self._branch_forward(name, inputs, data_samples)
            logits[name] = lg

        scores_cpu = {name: F.softmax(lg, dim=1).detach().cpu() for name, lg in logits.items()}


        if getattr(self, 'enable_logit_fusion', False):
            fusion_device = logits[self.branch_names[0]].device
            fused_logits = None
            for name in self.branch_names:
                weight = float(self.logit_fusion_weights.get(name, 0.0))
                temp = float(self.logit_fusion_temperatures.get(name, 1.0))
                if temp <= 0:
                    raise ValueError(f'Fusion temperature for {name} must be positive, got {temp}.')
                term = weight * (logits[name].to(fusion_device, non_blocking=True) / temp)
                fused_logits = term if fused_logits is None else fused_logits + term

            scores_cpu[self.logit_fusion_name] = F.softmax(
                fused_logits, dim=1).detach().cpu()

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
        if mode == 'predict':
            return self.predict(inputs, data_samples)

        out = {}
        for name in self.branch_names:
            lg, _, _, _ = self._branch_forward(name, inputs, data_samples)
            out[name] = lg
        return out
