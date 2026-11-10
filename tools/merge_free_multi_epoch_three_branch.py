from __future__ import print_function

import argparse
import copy
import json
import math
import os
from collections import defaultdict

import torch


BRANCHES = ("mv_frame", "mv_stream", "mv_voxel")
GROUPS = ("rgb", "event", "fusion", "head", "other", "private")


def log(msg):
    print(msg, flush=True)


def resolve_path(path):
    
    if path is None:
        return None
    if os.path.exists(path):
        return path

    prefix = "work2_ftkd2" + os.sep
    if path.startswith(prefix):
        stripped = path[len(prefix):]
        if os.path.exists(stripped):
            return stripped

    prefix2 = "." + os.sep + "work2_ftkd2" + os.sep
    if path.startswith(prefix2):
        stripped = path[len(prefix2):]
        if os.path.exists(stripped):
            return stripped
    return path


def load_checkpoint(path):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "ema_state_dict" in ckpt:
            state = ckpt["ema_state_dict"]
        else:

            state = ckpt
    else:
        raise TypeError("Unsupported checkpoint type from %s: %s" % (path, type(ckpt)))


    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return ckpt, out


def make_output_checkpoint(template_ckpt, out_state):
    if isinstance(template_ckpt, dict):
        out_ckpt = {}

        for k, v in template_ckpt.items():
            if k in ("optimizer", "optim_wrapper", "param_schedulers", "message_hub"):
                continue
            if k in ("state_dict", "model", "ema_state_dict"):
                continue
            out_ckpt[k] = v
        out_ckpt["state_dict"] = out_state
        meta = out_ckpt.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta["merged_by"] = "merge_free_multi_epoch_three_branch.py"
        out_ckpt["meta"] = meta
        return out_ckpt
    return {"state_dict": out_state}


def parse_csv_set(s):
    if s is None or s == "":
        return set()
    return set([x.strip() for x in s.split(",") if x.strip()])


def parse_branch_weights(s):
    
    if not s:
        return {b: 1.0 / len(BRANCHES) for b in BRANCHES}
    out = {}
    for item in s.split(","):
        if not item.strip():
            continue
        k, v = item.split("=")
        out[k.strip()] = float(v)
    for b in BRANCHES:
        out.setdefault(b, 0.0)
    return out


def get_branch_name(key):

    for b in BRANCHES:
        if ("." + b + ".") in key or key.startswith(b + ".") or ("/" + b + "/") in key:
            return b
    return None


def group_of_key(key):
    lk = key.lower()


    if "event_adapter" in lk:
        return "private"


    if "cls_head" in lk or ".head." in lk or lk.endswith(".head.weight") or lk.endswith(".head.bias"):
        return "head"


    fusion_words = [
        "former3", "former4", "res2former3", "res2former4",
        "conv_1x1_event_layer3", "conv_1x1_event_layer4",
        "fusion", ".token", "token."
    ]
    if any(w in lk for w in fusion_words):
        return "fusion"


    if "resnettsm_event" in lk or "event_backbone" in lk or ".event." in lk:
        return "event"


    if "resnettsm" in lk or "rgb" in lk or "image" in lk:
        return "rgb"

    return "other"


def is_float_tensor(x):
    return torch.is_tensor(x) and x.is_floating_point()


def center_hp_mask_2d(shape_hw, cutoff, device):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError("invalid fft shape: %r" % (shape_hw,))
    yy = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2.0
    xx = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0

    try:
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(yy, xx)

    denom_y = max((h - 1) / 2.0, 1.0)
    denom_x = max((w - 1) / 2.0, 1.0)
    radius = torch.sqrt((gy / denom_y) ** 2 + (gx / denom_x) ** 2)

    return (radius >= float(cutoff)).to(torch.float32)


def _as_2d_for_fft(x, mode):
    orig_shape = tuple(x.shape)
    if x.numel() <= 1:
        return x.reshape(1, 1), orig_shape
    if mode == "last2d" and x.dim() >= 2:
        return x.reshape(-1, x.shape[-2], x.shape[-1]).mean(dim=0), orig_shape

    if x.dim() >= 2:
        return x.reshape(x.shape[0], -1), orig_shape
    return x.reshape(1, -1), orig_shape


def fourier_split(delta, cutoff, mode="flatten2d"):
    
    if cutoff <= 0 or delta.numel() <= 1:
        high = delta.clone()
        low = torch.zeros_like(delta)
        return high, low

    dtype = delta.dtype
    x = delta.float()
    x2, orig_shape = _as_2d_for_fft(x, mode)

    if mode == "last2d":
        x2 = x.reshape(x.shape[0], -1) if x.dim() >= 2 else x.reshape(1, -1)

    spec = torch.fft.fft2(x2)
    spec_shift = torch.fft.fftshift(spec)
    mask = center_hp_mask_2d(spec_shift.shape[-2:], cutoff, spec_shift.device)
    high_shift = spec_shift * mask
    high2 = torch.fft.ifft2(torch.fft.ifftshift(high_shift)).real
    high = high2.reshape(orig_shape).to(dtype)
    low = (delta - high).to(dtype)
    return high, low


def sparse_topk(x, density, max_rescale=5.0, rescale="paper_abs"):
    
    if density <= 0 or x.numel() == 0:
        return torch.zeros_like(x)
    if density >= 1:
        return x.clone()

    flat = x.reshape(-1)
    k = max(1, int(round(flat.numel() * float(density))))
    k = min(k, flat.numel())
    _, idx = torch.topk(flat.abs(), k, largest=True, sorted=False)
    out_flat = torch.zeros_like(flat)
    out_flat[idx] = flat[idx]
    out = out_flat.reshape_as(x)

    if rescale == "none":
        return out
    eps = 1e-12
    if rescale in ("absmean", "paper_abs"):


        src_abs = x.abs().mean()
        dst_abs = out.abs().mean()
        scale = float((src_abs / (dst_abs + eps)).clamp(max=float(max_rescale)))
        return out * scale
    raise ValueError("Unknown expert rescale: %s" % rescale)


def normalize_weights(ws):
    vals = [max(float(w), 0.0) for w in ws]
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(vals) for _ in vals]
    return [v / s for v in vals]


def build_lambda_weights(deltas, base_weights, lambda_mode):
    if len(deltas) == 0:
        return []
    base = torch.tensor(normalize_weights(base_weights), dtype=torch.float32)
    if lambda_mode == "uniform":
        return base.tolist()

    stats = []
    for d in deltas:
        if d.numel() == 0:
            stats.append(0.0)
        else:
            stats.append(float(d.float().abs().mean()))
    stats_t = torch.tensor(stats, dtype=torch.float32)

    eps = 1e-12
    if lambda_mode == "absmean":
        score = base * (stats_t + eps)
    elif lambda_mode in ("inv_absmean", "free_absmean"):

        score = base / (stats_t + eps)
    elif lambda_mode == "softmax_absmean":
        score = torch.softmax(stats_t, dim=0) * base
    else:
        raise ValueError("Unknown lambda mode: %s" % lambda_mode)

    if float(score.sum()) <= 0:
        return base.tolist()
    return (score / score.sum()).tolist()


def weighted_sum(tensors, weights):
    assert len(tensors) == len(weights)
    out = torch.zeros_like(tensors[0].float())
    for t, w in zip(tensors, weights):
        out = out + float(w) * t.float()
    return out.to(tensors[0].dtype)


def get_group_arg(args, name, group):
    return getattr(args, "%s_%s" % (name, group))


def get_tensor_or_none(state, key):
    v = state.get(key, None)
    if torch.is_tensor(v):
        return v
    return None


def choose_anchor(args, key, branch, branch_anchor_state, specialist_state, candidate_items):
    if args.anchor_mode == "branch_anchor":
        t = get_tensor_or_none(branch_anchor_state, key)
        if t is not None:
            return t
    if args.anchor_mode == "specialist":
        t = get_tensor_or_none(specialist_state, key)
        if t is not None:
            return t
    if args.anchor_mode == "mean_candidates":
        vals = [get_tensor_or_none(st, key) for _, st, _ in candidate_items]
        vals = [v for v in vals if v is not None and is_float_tensor(v)]
        if vals:
            return weighted_sum(vals, [1.0 / len(vals)] * len(vals))

    t = get_tensor_or_none(branch_anchor_state, key)
    if t is not None:
        return t
    t = get_tensor_or_none(specialist_state, key)
    if t is not None:
        return t
    for _, st, _ in candidate_items:
        t = get_tensor_or_none(st, key)
        if t is not None:
            return t
    return None


def merge_one_tensor(args, key, group, branch, anchor, specialist, candidate_items, report):
    
    if specialist is None:

        specialist = anchor

    if anchor is None or specialist is None:
        report["missing"] += 1
        return None

    if (not is_float_tensor(specialist)) or (not is_float_tensor(anchor)):
        report["non_float"] += 1
        return specialist.clone()

    if tuple(specialist.shape) != tuple(anchor.shape):
        report["shape_mismatch"] += 1
        return specialist.clone()


    cand_tensors, cand_names, cand_base_weights = [], [], []
    for name, st, bw in candidate_items:
        t = get_tensor_or_none(st, key)
        if t is None:
            continue
        if not is_float_tensor(t):
            continue
        if tuple(t.shape) != tuple(anchor.shape):
            continue
        cand_tensors.append(t)
        cand_names.append(name)
        cand_base_weights.append(float(bw))

    if len(cand_tensors) == 0:
        report["no_candidate"] += 1
        return specialist.clone()

    if args.method == "assemble":
        return specialist.clone()

    if args.method == "soup":

        sw = normalize_weights(cand_base_weights)
        raw = weighted_sum(cand_tensors, sw)
        keep = get_group_arg(args, "keep", group)
        final = float(keep) * specialist.float() + (1.0 - float(keep)) * raw.float()
        report["merged"][group] += 1
        return final.to(specialist.dtype)


    deltas = [(t.float() - anchor.float()).to(torch.float32) for t in cand_tensors]
    cutoff = get_group_arg(args, "cutoff", group)
    high_parts, low_parts = [], []
    for d in deltas:
        hp, lp = fourier_split(d, cutoff, args.fft_mode)
        high_parts.append(hp.float())
        low_parts.append(lp.float())

    lambdas = build_lambda_weights(high_parts, cand_base_weights, args.lambda_mode)
    merged_high = weighted_sum(high_parts, lambdas).float()
    raw = anchor.float() + float(get_group_arg(args, "delta_scale", group)) * merged_high

    if args.method == "free":

        if args.expert_source == "low":
            expert_pool = low_parts
        elif args.expert_source == "high":
            expert_pool = high_parts
        elif args.expert_source == "delta":
            expert_pool = deltas
        else:
            raise ValueError("Unknown expert source: %s" % args.expert_source)

        merged_expert_dense = weighted_sum(expert_pool, lambdas).float()
        density = get_group_arg(args, "expert_density", group)
        expert = sparse_topk(
            merged_expert_dense,
            density=density,
            max_rescale=args.expert_max_rescale,
            rescale=args.expert_rescale,
        ).float()
        raw = raw + float(get_group_arg(args, "expert_scale", group)) * expert

    keep = get_group_arg(args, "keep", group)
    final = float(keep) * specialist.float() + (1.0 - float(keep)) * raw.float()
    report["merged"][group] += 1
    if args.verbose and report["merged"][group] <= 3:
        log("[MERGE] %s | branch=%s group=%s source=%s lambdas=%s" % (
            key, branch, group, ",".join(cand_names),
            ",".join(["%.3f" % x for x in lambdas]),
        ))
    return final.to(specialist.dtype)


def add_group_args(parser):

    defaults = {
        "rgb":    dict(cutoff=0.03, expert_density=0.0,   expert_scale=0.0,  delta_scale=1.0, keep=1.0),
        "event":  dict(cutoff=0.03, expert_density=0.002, expert_scale=0.05, delta_scale=1.0, keep=1.0),
        "fusion": dict(cutoff=0.03, expert_density=0.005, expert_scale=0.10, delta_scale=1.0, keep=0.995),
        "head":   dict(cutoff=0.03, expert_density=0.010, expert_scale=0.25, delta_scale=1.0, keep=0.990),
        "other":  dict(cutoff=0.03, expert_density=0.0,   expert_scale=0.0,  delta_scale=1.0, keep=1.0),
    }
    for g, vals in defaults.items():
        parser.add_argument("--cutoff-%s" % g, dest="cutoff_%s" % g, type=float, default=vals["cutoff"])
        parser.add_argument("--expert-density-%s" % g, dest="expert_density_%s" % g, type=float, default=vals["expert_density"])
        parser.add_argument("--expert-scale-%s" % g, dest="expert_scale_%s" % g, type=float, default=vals["expert_scale"])
        parser.add_argument("--delta-scale-%s" % g, dest="delta_scale_%s" % g, type=float, default=vals["delta_scale"])
        parser.add_argument("--keep-%s" % g, dest="keep_%s" % g, type=float, default=vals["keep"])


def build_argparser():
    parser = argparse.ArgumentParser(
        description="FREE/FR-style multi-epoch full three-branch checkpoint merger"
    )
    parser.add_argument("--template-ckpt", required=True,
                        help="A full three-branch checkpoint used as output template/meta.")
    parser.add_argument("--out", required=True)


    parser.add_argument("--frame-anchor-ckpt", required=True)
    parser.add_argument("--stream-anchor-ckpt", required=True)
    parser.add_argument("--voxel-anchor-ckpt", required=True)


    parser.add_argument("--frame-specialist-ckpt", required=True)
    parser.add_argument("--stream-specialist-ckpt", required=True)
    parser.add_argument("--voxel-specialist-ckpt", required=True)


    parser.add_argument("--epoch-ckpts", nargs="*", default=[],
                        help="Full three-branch checkpoints, e.g. epoch_10.pth ... epoch_30.pth")

    parser.add_argument("--method", choices=["free", "fr", "soup", "assemble"], default="free")
    parser.add_argument("--merge-groups", default="head,fusion",
                        help="Comma-separated groups among rgb,event,fusion,head,other. private is never merged unless --merge-private.")
    parser.add_argument("--merge-private", action="store_true",
                        help="Not recommended. Allows event_adapter/private group to be merged.")
    parser.add_argument("--anchor-mode", choices=["branch_anchor", "specialist", "mean_candidates"],
                        default="branch_anchor")
    parser.add_argument("--fft-mode", choices=["flatten2d", "last2d"], default="flatten2d")
    parser.add_argument("--lambda-mode", choices=["uniform", "absmean", "inv_absmean", "free_absmean", "softmax_absmean"],
                        default="uniform")
    parser.add_argument("--specialist-weight", type=float, default=1.0)
    parser.add_argument("--epoch-weight", type=float, default=0.20,
                        help="Base weight for each epoch checkpoint before normalization.")
    parser.add_argument("--epoch-decay", type=float, default=1.0,
                        help="Optional decay applied to epoch sources in listed order. 1.0 means no decay.")
    parser.add_argument("--include-anchor-as-source", action="store_true",
                        help="Include mv_mutual best branch checkpoint as an additional source.")
    parser.add_argument("--expert-source", choices=["low", "high", "delta"], default="low")
    parser.add_argument("--expert-rescale", choices=["none", "absmean", "paper_abs"], default="paper_abs")
    parser.add_argument("--expert-max-rescale", type=float, default=5.0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--verbose", action="store_true")
    add_group_args(parser)
    return parser


def main():
    args = build_argparser().parse_args()

    merge_groups = parse_csv_set(args.merge_groups)
    bad = merge_groups - set(GROUPS)
    if bad:
        raise ValueError("Unknown merge groups: %s" % sorted(bad))
    if "private" in merge_groups and not args.merge_private:
        log("[WARN] private requested but --merge-private is not set; private group will be kept from specialists.")
        merge_groups.discard("private")

    log("[INFO] Loading template: %s" % resolve_path(args.template_ckpt))
    template_ckpt, template_state = load_checkpoint(args.template_ckpt)
    out_state = copy.deepcopy(template_state)

    anchor_paths = {
        "mv_frame": args.frame_anchor_ckpt,
        "mv_stream": args.stream_anchor_ckpt,
        "mv_voxel": args.voxel_anchor_ckpt,
    }
    specialist_paths = {
        "mv_frame": args.frame_specialist_ckpt,
        "mv_stream": args.stream_specialist_ckpt,
        "mv_voxel": args.voxel_specialist_ckpt,
    }

    log("[INFO] Loading branch anchors...")
    anchor_states = {}
    for b in BRANCHES:
        _, st = load_checkpoint(anchor_paths[b])
        anchor_states[b] = st
        log("  - %s anchor: %s" % (b, resolve_path(anchor_paths[b])))

    log("[INFO] Loading branch specialists...")
    specialist_states = {}
    for b in BRANCHES:
        _, st = load_checkpoint(specialist_paths[b])
        specialist_states[b] = st
        log("  - %s specialist: %s" % (b, resolve_path(specialist_paths[b])))

    log("[INFO] Loading extra epoch checkpoints...")
    epoch_states = []
    for i, p in enumerate(args.epoch_ckpts):
        _, st = load_checkpoint(p)
        decay = float(args.epoch_decay) ** i
        epoch_states.append((os.path.basename(p), st, float(args.epoch_weight) * decay))
        log("  - epoch source: %s weight=%.4f" % (resolve_path(p), float(args.epoch_weight) * decay))

    report = {
        "method": args.method,
        "merge_groups": sorted(list(merge_groups)),
        "anchor_mode": args.anchor_mode,
        "merged": defaultdict(int),
        "kept_private": 0,
        "kept_group": defaultdict(int),
        "missing": 0,
        "non_float": 0,
        "shape_mismatch": 0,
        "no_candidate": 0,
        "total_branch_keys": 0,
    }

    for key, old_value in template_state.items():
        branch = get_branch_name(key)
        if branch is None:

            continue

        group = group_of_key(key)
        report["total_branch_keys"] += 1

        specialist_state = specialist_states[branch]
        branch_anchor_state = anchor_states[branch]
        specialist_tensor = get_tensor_or_none(specialist_state, key)
        anchor_tensor = get_tensor_or_none(branch_anchor_state, key)


        if specialist_tensor is not None:
            out_state[key] = specialist_tensor.clone()
        elif not args.allow_missing:
            raise KeyError("Specialist checkpoint for %s lacks key %s" % (branch, key))

        if group == "private" and not args.merge_private:
            report["kept_private"] += 1
            continue

        if group not in merge_groups:
            report["kept_group"][group] += 1
            continue

        candidate_items = []
        if args.include_anchor_as_source:
            candidate_items.append(("anchor_%s" % branch, branch_anchor_state, args.epoch_weight))
        candidate_items.append(("specialist_%s" % branch, specialist_state, args.specialist_weight))
        for name, st, w in epoch_states:
            candidate_items.append((name, st, w))

        anchor = choose_anchor(args, key, branch, branch_anchor_state, specialist_state, candidate_items)
        merged = merge_one_tensor(
            args=args,
            key=key,
            group=group,
            branch=branch,
            anchor=anchor,
            specialist=specialist_tensor,
            candidate_items=candidate_items,
            report=report,
        )
        if merged is not None:
            out_state[key] = merged.clone()
        elif not args.allow_missing:
            raise RuntimeError("Failed to merge key: %s" % key)


    report["merged"] = dict(report["merged"])
    report["kept_group"] = dict(report["kept_group"])

    log("[INFO] Merge summary:")
    for k, v in report.items():
        if k in ("merged", "kept_group"):
            log("  %s: %s" % (k, json.dumps(v, ensure_ascii=False, sort_keys=True)))
        else:
            log("  %s: %s" % (k, v))

    if args.dry_run:
        log("[DRY-RUN] Not saving checkpoint.")
        return

    out_path = resolve_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_ckpt = make_output_checkpoint(template_ckpt, out_state)
    torch.save(out_ckpt, out_path)
    log("[OK] Saved merged full three-branch checkpoint: %s" % out_path)

    report_path = args.report_json
    if report_path is None:
        report_path = out_path + ".merge_report.json"
    report_path = resolve_path(report_path)
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    log("[OK] Saved report: %s" % report_path)


if __name__ == "__main__":
    main()
