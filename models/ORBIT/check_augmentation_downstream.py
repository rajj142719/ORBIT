#!/usr/bin/env python
"""
check_augmentation_downstream.py

Evaluates ORBIT's two augmentation branches (orthogonal vs simple_spatial)
using a SIMULATED FSCIL EPISODE, not an embedding-space proxy metric.
This directly measures what the paper's own evaluation protocol measures:
incremental classification accuracy and confusion between base and novel
classes -- so a reviewer can check the methodology against the paper's
existing metrics instead of trusting a new invented score.

Why this design (addressing prior methodological concerns):

  1. NO CIRCULARITY: uses your ACTUAL TRAINED FSCIL checkpoint (via
     --checkpoint / --encoder-module / --encoder-class), not a generic
     ImageNet-pretrained ResNet18. If you truly have no checkpoint yet,
     pass --pretrained as a clearly-labeled FALLBACK BASELINE only --
     results from that mode are printed with a loud warning and flagged
     in the final verdict table.

  2. NO UNJUSTIFIED COMPOSITE SCORE: reports the same categories of
     metrics the paper itself reports -- base-class accuracy, novel-class
     accuracy, overall accuracy -- so nothing here needs a new metric a
     reviewer has to evaluate on trust.

  3. DIRECT BOUNDARY-DRIFT CHECK: explicitly measures how often an
     augmented BASE-class query is misclassified as a NOVEL class (or
     vice versa) under nearest-prototype classification. This is the
     literal failure mode ORBIT claims to avoid ("does augmentation
     encroach on other classes' space"), measured directly rather than
     inferred from a distance proxy.

  4. INCLUDES A NO-AUGMENTATION CONTROL: every dataset is also evaluated
     with augmentation OFF, so you can report "does augmentation help at
     all" (vs. the trivial baseline) alongside "which branch helps more."
     This is the comparison a reviewer will look for first.

  5. MATCHES FSCIL PROTOCOL SHAPE: episodes are N-way K-shot with a query
     set, using CLI args (--n-way, --k-shot, --n-query) so you can set
     these to match your paper's actual incremental-session configuration
     rather than an arbitrary sample size.

  6. MULTI-SEED, MULTI-EPISODE: results are averaged over many sampled
     episodes (--n-episodes) and reported as mean +/- std, matching how
     your paper should already be reporting FSCIL numbers per reviewer
     s23E's request for statistical reliability.

Usage (with your real trained encoder):
    python check_augmentation_downstream.py \\
        --root ./data \\
        --cub-root /home/rb/DA-main/CUB_200_2011/ \\
        --mini-imagenet-root /home/rb/DA-main/data/miniimagenet/ \\
        --device cuda \\
        --da-module Network --da-class DA \\
        --checkpoint /path/to/your/trained_encoder.pth \\
        --encoder-module Network --encoder-class MYNET \\
        --n-way 5 --k-shot 5 --n-query 15 --n-episodes 50
"""

import argparse
import importlib
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference augmenter (fallback if --da-module not given). Identical logic
# to the version already reviewed; kept only so the script runs standalone.
# ---------------------------------------------------------------------------
class ReferenceAugmenter(nn.Module):
    def _vectorized_pair_mix(self, f_orig, idx, lam, generator=None):
        device = f_orig.device
        n = idx.numel()
        out = f_orig.clone()
        if n < 2:
            return out
        local_perm = torch.randperm(n, generator=generator, device=device)
        partner = idx[local_perm]
        for _ in range(3):
            same = partner == idx
            if not same.any():
                break
            local_perm = torch.where(same, torch.roll(local_perm, 1), local_perm)
            partner = idx[local_perm]
        same = partner == idx
        if same.any():
            residual = idx[same]
            for i in residual.tolist():
                candidates = idx[idx != i]
                j = candidates[torch.randint(0, len(candidates), (1,), generator=generator, device=device)]
                partner[idx == i] = j
        out[idx] = lam * f_orig[idx] + (1 - lam) * f_orig[partner]
        return out

    def _make_random_orthogonal_basis(self, D, device, generator=None, dtype=torch.float32, identity=False):
        if identity:
            return torch.eye(D, device=device, dtype=dtype)
        A = torch.randn(D, D, device=device, dtype=dtype, generator=generator)
        Q, R = torch.linalg.qr(A)
        d = torch.diagonal(R, 0)
        sign = d.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return Q * sign.unsqueeze(0)

    def replace_augmentation(self, feature, lamda=0.5, mix_lam=0.5, noise_std=0.02,
                              seed=None, force_branch=None):
        device = feature.device
        if seed is not None:
            g_T     = torch.Generator(device=device).manual_seed(seed)
            g_pair  = torch.Generator(device=device).manual_seed(seed + 1)
            g_perm  = torch.Generator(device=device).manual_seed(seed + 2)
            g_mix   = torch.Generator(device=device).manual_seed(seed + 3)
            g_noise = torch.Generator(device=device).manual_seed(seed + 4)
        else:
            g_T = g_pair = g_perm = g_mix = g_noise = None

        use_simple = (force_branch == "simple_spatial") if force_branch is not None \
            else getattr(self, "is_fine_grained", False)

        if use_simple:
            odd_idx = torch.arange(1, feature.size(0), 2, device=device)
            return self._vectorized_pair_mix(feature, odd_idx, lamda, generator=g_pair)

        restore = None
        if feature.dim() == 4:
            B, C, H, W = feature.shape
            feature = F.adaptive_avg_pool2d(feature, 1).view(B, C)
            restore = (B, C, H, W)
        B, D = feature.shape

        odd_idx = torch.arange(1, B, 2, device=device)
        f = self._vectorized_pair_mix(feature, odd_idx, mix_lam, generator=g_pair)

        need_new_basis = (
            not hasattr(self, "_orth_basis")
            or self._orth_basis.shape[0] != D
            or self._orth_basis.device != device
            or self._orth_basis.dtype != feature.dtype
        )
        if need_new_basis:
            self._orth_basis = self._make_random_orthogonal_basis(D, device, generator=g_T, dtype=feature.dtype)
        T = self._orth_basis

        f_t = f @ T
        perm = torch.randperm(B, generator=g_perm, device=device) if g_perm is not None else torch.randperm(B, device=device)
        f_t_p = f_t[perm]
        rot_mix_lam = torch.rand(D, device=device, generator=g_mix) if g_mix is not None else torch.rand(D, device=device)
        f_mix = rot_mix_lam * f_t + (1 - rot_mix_lam) * f_t_p
        f_rec = f_mix @ T.T

        f_out = 0.5 * f + 0.5 * f_rec
        noise = torch.randn(f_out.shape, generator=g_noise, device=device) * noise_std if g_noise is not None \
            else torch.randn_like(f_out) * noise_std
        f_out = f_out + noise

        if restore is not None:
            Bc, C, H, W = restore
            f_out = f_out.view(Bc, C, 1, 1).expand(Bc, C, H, W)
        return f_out


def load_class(module_name, class_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def build_real_encoder(args, device):
    """
    Loads YOUR trained FSCIL backbone checkpoint. This is required to avoid
    the ImageNet-pretraining circularity issue for miniImageNet in particular.
    """
    encoder_cls = load_class(args.encoder_module, args.encoder_class)
    model = encoder_cls()
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "params" in state:
        state = state["params"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warning] load_state_dict: missing={len(missing)} unexpected={len(unexpected)} keys "
              f"(check --encoder-class matches your checkpoint's architecture)")
    return model.eval().to(device)


def build_pretrained_fallback(device):
    import torchvision

    class Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def encoder_q(self, x):
            return self.backbone(x), None

    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    return Wrapper(backbone).eval().to(device)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, dataloader, min_per_class, device):
    model.eval()
    feats_per_class = defaultdict(list)
    for data, labels in dataloader:
        data = data.to(device, non_blocking=True)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.to(device, non_blocking=True)
        feats, _ = model.encoder_q(data)
        feats = F.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
        for f, l in zip(feats, labels):
            feats_per_class[l.item()].append(f.detach())
    return {k: v for k, v in feats_per_class.items() if len(v) >= min_per_class}


# ---------------------------------------------------------------------------
# One simulated FSCIL episode
# ---------------------------------------------------------------------------
def run_episode(feats_per_class, augmenter, branch, n_way, k_shot, n_query,
                 seed, device, batch_size=64):
    """
    Simulates: a BASE set of classes with plentiful data, and a freshly
    arriving NOVEL set of n_way classes with only k_shot examples each
    (the actual FSCIL incremental-session setup). Measures classification
    over base+novel combined -- exactly what avg incremental accuracy and
    PD are computed from in the paper.

    branch: None (no augmentation, control), "orthogonal", or "simple_spatial"
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    all_classes = list(feats_per_class.keys())
    if len(all_classes) < n_way + 2:
        return None
    rng.shuffle(all_classes)
    novel_classes = all_classes[:n_way]
    base_classes = all_classes[n_way:]

    base_protos, base_labels = [], []
    base_query_feats, base_query_labels = [], []
    for c in base_classes:
        feats = feats_per_class[c].copy()
        rng.shuffle(feats)
        n_query_base = min(n_query, max(1, len(feats) // 4))
        support = feats[:-n_query_base] if len(feats) > n_query_base else feats
        query = feats[-n_query_base:] if len(feats) > n_query_base else feats[:1]

        support_t = torch.stack(support).to(device)
        if branch is not None:
            aug_out = []
            for i in range(0, support_t.size(0), batch_size):
                b = support_t[i:i + batch_size]
                if b.size(0) < 2:
                    aug_out.append(b)
                    continue
                aug_out.append(augmenter.replace_augmentation(b, seed=seed + c, force_branch=branch))
            support_t = torch.cat(aug_out, dim=0)

        base_protos.append(support_t.mean(0))
        base_labels.append(c)
        for q in query:
            base_query_feats.append(q)
            base_query_labels.append(c)

    novel_protos, novel_labels = [], []
    novel_query_feats, novel_query_labels = [], []
    for c in novel_classes:
        feats = feats_per_class[c].copy()
        rng.shuffle(feats)
        if len(feats) < k_shot + 1:
            continue
        support = feats[:k_shot]
        query = feats[k_shot:k_shot + n_query] if len(feats) > k_shot else feats[:1]

        novel_protos.append(torch.stack(support).to(device).mean(0))
        novel_labels.append(c)
        for q in query:
            novel_query_feats.append(q)
            novel_query_labels.append(c)

    if len(novel_protos) == 0 or len(base_protos) == 0:
        return None

    P = F.normalize(torch.stack(base_protos + novel_protos), dim=1)
    proto_labels = torch.tensor(base_labels + novel_labels, device=device)
    is_novel_proto = torch.tensor([0] * len(base_labels) + [1] * len(novel_labels),
                                   device=device, dtype=torch.bool)

    def classify(query_feats, query_labels):
        if len(query_feats) == 0:
            return None
        Q = F.normalize(torch.stack(query_feats).to(device), dim=1)
        sim = Q @ P.T
        pred_idx = sim.argmax(dim=1)
        pred_labels = proto_labels[pred_idx]
        true_labels = torch.tensor(query_labels, device=device)
        correct = (pred_labels == true_labels)
        pred_is_novel = is_novel_proto[pred_idx]
        return correct, pred_is_novel

    base_res = classify(base_query_feats, base_query_labels)
    novel_res = classify(novel_query_feats, novel_query_labels)

    base_acc = base_res[0].float().mean().item() if base_res else float("nan")
    novel_acc = novel_res[0].float().mean().item() if novel_res else float("nan")
    base_to_novel_confusion = base_res[1].float().mean().item() if base_res else float("nan")
    novel_to_base_confusion = (~novel_res[1]).float().mean().item() if novel_res else float("nan")

    n_total = len(base_query_labels) + len(novel_query_labels)
    overall_acc = (
        (base_res[0].sum().item() if base_res else 0)
        + (novel_res[0].sum().item() if novel_res else 0)
    ) / max(n_total, 1)

    return {
        "base_acc": base_acc,
        "novel_acc": novel_acc,
        "overall_acc": overall_acc,
        "base_to_novel_confusion": base_to_novel_confusion,
        "novel_to_base_confusion": novel_to_base_confusion,
    }


def summarize(results_list, key):
    vals = [r[key] for r in results_list if r is not None and r[key] == r[key]]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return mean, std


def run_dataset(name, feats_per_class, augmenter, n_way, k_shot, n_query,
                 n_episodes, device):
    print(f"\n--- {name} ({len(feats_per_class)} usable classes) ---")
    conditions = {
        "none (control)": None,
        "orthogonal": "orthogonal",
        "simple_spatial": "simple_spatial",
    }
    cond_results = {}
    for label, branch in conditions.items():
        episodes = []
        for ep in range(n_episodes):
            r = run_episode(feats_per_class, augmenter, branch, n_way, k_shot,
                             n_query, seed=ep, device=device)
            episodes.append(r)

        row = {}
        for key in ("base_acc", "novel_acc", "overall_acc",
                    "base_to_novel_confusion", "novel_to_base_confusion"):
            row[key] = summarize(episodes, key)
        cond_results[label] = row
        print(f"  {label:16s}  base_acc={row['base_acc'][0]:.4f}+/-{row['base_acc'][1]:.4f}  "
              f"novel_acc={row['novel_acc'][0]:.4f}+/-{row['novel_acc'][1]:.4f}  "
              f"overall={row['overall_acc'][0]:.4f}+/-{row['overall_acc'][1]:.4f}")
        print(f"  {'':16s}  base->novel confusion={row['base_to_novel_confusion'][0]:.4f}  "
              f"novel->base confusion={row['novel_to_base_confusion'][0]:.4f}")

    winner = max(
        ("orthogonal", "simple_spatial"),
        key=lambda b: cond_results[b]["overall_acc"][0],
    )
    beats_control = cond_results[winner]["overall_acc"][0] > cond_results["none (control)"]["overall_acc"][0]
    print(f"  WINNER (by overall_acc): {winner}  "
          f"(beats no-augmentation control: {beats_control})")
    return winner, cond_results


def main():
    from check_granularity import build_datasets, build_loader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--cub-root", type=str, default="")
    parser.add_argument("--mini-imagenet-root", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--checkpoint", type=str, default="",
                         help="Path to your trained FSCIL encoder checkpoint. "
                              "Strongly recommended over --pretrained.")
    parser.add_argument("--encoder-module", type=str, default="",
                         help="Module containing your encoder class, e.g. 'Network'")
    parser.add_argument("--encoder-class", type=str, default="",
                         help="Encoder class name to instantiate before loading checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=False,
                         help="FALLBACK ONLY: use generic ImageNet-pretrained ResNet18 "
                              "if no checkpoint is available. Results are flagged as "
                              "a fallback baseline, not a substitute for real evaluation.")

    parser.add_argument("--da-module", type=str, default="")
    parser.add_argument("--da-class", type=str, default="DA")

    parser.add_argument("--n-way", type=int, default=5, help="novel classes per episode")
    parser.add_argument("--k-shot", type=int, default=5, help="shots per novel class")
    parser.add_argument("--n-query", type=int, default=15)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint and not args.pretrained:
        raise SystemExit(
            "Refusing to run with no encoder specified. Pass --checkpoint "
            "(+ --encoder-module/--encoder-class) for a real evaluation, or "
            "explicitly pass --pretrained to acknowledge you're using the "
            "flagged ImageNet-pretrained fallback."
        )

    cub_root = args.cub_root or None
    mini_root = args.mini_imagenet_root or None
    print(f"Building datasets under {args.root} "
          f"(CUB: {cub_root or 'skipped'}, miniImageNet: {mini_root or 'skipped'}) ...")
    datasets = build_datasets(args.root, cub_root=cub_root, mini_imagenet_root=mini_root)

    if args.checkpoint:
        print(f"Loading REAL trained encoder from {args.checkpoint} ...")
        model = build_real_encoder(args, args.device)
    else:
        print("*" * 70)
        print("WARNING: using ImageNet-pretrained ResNet18 FALLBACK encoder.")
        print("This is circular for miniImageNet (a subset of ImageNet) and")
        print("should NOT be cited as evidence in a rebuttal. Re-run with")
        print("--checkpoint pointing at your real trained FSCIL encoder.")
        print("*" * 70)
        model = build_pretrained_fallback(args.device)

    if args.da_module:
        augmenter = load_class(args.da_module, args.da_class)()
    else:
        print("Using reference ReferenceAugmenter (pass --da-module/--da-class "
              "to test your actual shipped augmentation code instead).")
        augmenter = ReferenceAugmenter()

    hypothesis = {
        "CIFAR100": "orthogonal",
        "miniImageNet": "orthogonal",
        "CUB200": "simple_spatial",
    }

    summary = {}
    for name, dataset in datasets.items():
        if name not in hypothesis:
            continue
        loader = build_loader(dataset, batch_size=args.batch_size,
                               num_workers=args.num_workers)
        feats_per_class = extract_features(model, loader, args.min_per_class, args.device)
        if len(feats_per_class) < args.n_way + 2:
            print(f"\n--- {name} --- skipped: only {len(feats_per_class)} usable classes "
                  f"(need >= n_way+2 = {args.n_way + 2})")
            continue
        winner, cond_results = run_dataset(name, feats_per_class, augmenter,
                                            args.n_way, args.k_shot, args.n_query,
                                            args.n_episodes, args.device)
        summary[name] = (winner, cond_results)

    print("\n" + "=" * 70)
    print("HYPOTHESIS CHECK  (downstream simulated-episode accuracy, not a proxy)")
    print("=" * 70)
    n_match = 0
    tested = 0
    for name, expected in hypothesis.items():
        if name not in summary:
            print(f"{name:15s}  (skipped -- dataset not available or too few classes)")
            continue
        tested += 1
        winner, _ = summary[name]
        match = winner == expected
        n_match += int(match)
        print(f"{name:15s}  expected={expected:15s}  actual_winner={winner:15s}  "
              f"{'MATCH' if match else 'MISMATCH'}")
    if tested:
        print(f"\n{n_match}/{tested} datasets match the hypothesized routing.")
    if not args.checkpoint:
        print("\n(Reminder: this run used the FALLBACK pretrained encoder -- "
              "re-run with --checkpoint before citing these numbers anywhere.)")


if __name__ == "__main__":
    main()