# (paste check_datasets_fine_grained.py content here)
#!/usr/bin/env python
"""
check_datasets_fine_grained.py  (a.k.a. test_check_granularity.py)

Test harness for check_granularity.py's check_fine_grained().

Two modes:
  1. --synthetic (default, no GPU/downloads needed):
     Builds synthetic "fake datasets" with KNOWN, controllable separation
     (via a toy encoder that returns feature vectors directly), to verify
     the accuracy/ratio metric behaves correctly:
       - well-separated clusters -> high ratio -> "coarse"
       - overlapping/noisy clusters -> low ratio -> "fine"
       - sanity checks on n_seeds stability, support_ratio, edge cases
     This validates the METRIC LOGIC is correct before you trust it on
     real data -- i.e. it answers "does this code do what it claims to do."

  2. --real:
     Runs check_fine_grained() from check_granularity.py against your
     actual dataset roots and a real (or pretrained) encoder, and reports
     agreement with literature ground truth. This is the calibration run
     for the rebuttal.

Usage:
    python check_datasets_fine_grained.py --synthetic
    python check_datasets_fine_grained.py --real --root ./data \\
        --cub-root /home/rb/DA-main/CUB_200_2011 \\
        --mini-imagenet-root /home/rb/DA-main/data/miniimagenet
"""

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import the metric under test directly from your existing file.
# check_granularity.py must be on the path (same directory).
from check_granularity import check_fine_grained, GROUND_TRUTH


# ---------------------------------------------------------------------------
# Synthetic data + toy encoder for controlled unit tests
# ---------------------------------------------------------------------------
class SyntheticClusterDataset(Dataset):
    """
    Generates `num_classes` Gaussian clusters in `dim`-dimensional space.
    `cluster_std` controls overlap: small std -> well separated (coarse-like),
    large std -> overlapping (fine-like).
    Class centers are placed on the unit sphere at fixed, evenly-spaced
    angles so separation is controllable and reproducible.
    """

    def __init__(self, num_classes=20, samples_per_class=60, dim=64,
                 cluster_std=0.05, seed=0):
        g = torch.Generator().manual_seed(seed)
        centers = torch.randn(num_classes, dim, generator=g)
        centers = centers / centers.norm(dim=1, keepdim=True)
        self.centers = centers

        feats, labels = [], []
        for c in range(num_classes):
            noise = torch.randn(samples_per_class, dim, generator=g) * cluster_std
            feats.append(centers[c].unsqueeze(0) + noise)
            labels.extend([c] * samples_per_class)
        self.feats = torch.cat(feats, dim=0)
        self.labels = torch.tensor(labels)

        perm = torch.randperm(len(self.labels), generator=g)
        self.feats = self.feats[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


class IdentityEncoderQWrapper(nn.Module):
    """
    Toy 'model' whose encoder_q just returns the input feature vector
    unchanged, reshaped to look like a conv feature map [B, C, 1, 1] so it
    survives check_fine_grained's adaptive_avg_pool2d call unchanged.
    """
    def encoder_q(self, x):
        return x.unsqueeze(-1).unsqueeze(-1), None


def make_synthetic_loader(num_classes, samples_per_class, dim, cluster_std,
                           seed=0, batch_size=64):
    ds = SyntheticClusterDataset(
        num_classes=num_classes, samples_per_class=samples_per_class,
        dim=dim, cluster_std=cluster_std, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


# ---------------------------------------------------------------------------
# Synthetic test suite
# ---------------------------------------------------------------------------
def run_synthetic_tests(device="cpu", verbose=True):
    model = IdentityEncoderQWrapper().to(device)
    results = []
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {name}  {detail}")
        results.append((name, condition, detail))

    print("=" * 70)
    print("TEST 1: Well-separated clusters -> should be predicted COARSE")
    print("=" * 70)
    loader = make_synthetic_loader(
        num_classes=20, samples_per_class=60, dim=64,
        cluster_std=0.02, seed=1,
    )
    is_fine, acc, chance, ratio = check_fine_grained(
        model, loader, ratio_threshold=3.0, max_classes=20,
        max_per_class=60, support_ratio=0.5, seed=0, device=device,
    )
    check("tight clusters classify near-perfectly", acc > 0.95, f"acc={acc:.4f}")
    check("tight clusters routed as coarse (is_fine=False)", is_fine is False,
          f"ratio={ratio:.2f}x threshold=3.0")

    print()
    print("=" * 70)
    print("TEST 2: Heavily overlapping clusters -> should be predicted FINE")
    print("=" * 70)
    loader = make_synthetic_loader(
        num_classes=20, samples_per_class=60, dim=64,
        cluster_std=1.5, seed=2,
    )
    is_fine, acc, chance, ratio = check_fine_grained(
        model, loader, ratio_threshold=3.0, max_classes=20,
        max_per_class=60, support_ratio=0.5, seed=0, device=device,
    )
    check("overlapping clusters classify near chance", acc < 2 * chance,
          f"acc={acc:.4f} chance={chance:.4f}")
    check("overlapping clusters routed as fine (is_fine=True)", is_fine is True,
          f"ratio={ratio:.2f}x threshold=3.0")

    print()
    print("=" * 70)
    print("TEST 3: Monotonicity -- ratio should decrease as cluster_std rises")
    print("=" * 70)
    stds = [0.02, 0.1, 0.3, 0.6, 1.2, 2.0]
    ratios = []
    for s in stds:
        loader = make_synthetic_loader(
            num_classes=20, samples_per_class=60, dim=64,
            cluster_std=s, seed=3,
        )
        _, acc, chance, ratio = check_fine_grained(
            model, loader, ratio_threshold=3.0, max_classes=20,
            max_per_class=60, support_ratio=0.5, seed=0, device=device,
        )
        ratios.append(ratio)
        print(f"  cluster_std={s:>5.2f}  acc={acc:.4f}  ratio={ratio:6.2f}x")
    is_monotonic = all(ratios[i] >= ratios[i + 1] - 1e-6 for i in range(len(ratios) - 1))
    check("ratio is (roughly) monotonically decreasing as overlap increases",
          is_monotonic, f"ratios={[round(r, 2) for r in ratios]}")

    print()
    print("=" * 70)
    print("TEST 4: Order-independence -- shuffled vs. sorted-by-class loader")
    print("=" * 70)
    ds = SyntheticClusterDataset(num_classes=30, samples_per_class=40,
                                  dim=64, cluster_std=0.3, seed=4)
    shuffled_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    sort_idx = torch.argsort(ds.labels)
    sorted_feats = ds.feats[sort_idx]
    sorted_labels = ds.labels[sort_idx]

    class SortedView(Dataset):
        def __len__(self): return len(sorted_labels)
        def __getitem__(self, i): return sorted_feats[i], sorted_labels[i]

    sorted_loader = DataLoader(SortedView(), batch_size=64, shuffle=False, num_workers=0)

    _, acc_shuf, _, ratio_shuf = check_fine_grained(
        model, shuffled_loader, ratio_threshold=3.0, max_classes=15,
        max_per_class=40, support_ratio=0.5, seed=0, device=device,
    )
    _, acc_sort, _, ratio_sort = check_fine_grained(
        model, sorted_loader, ratio_threshold=3.0, max_classes=15,
        max_per_class=40, support_ratio=0.5, seed=0, device=device,
    )
    rel_diff = abs(ratio_shuf - ratio_sort) / max(ratio_shuf, ratio_sort, 1e-6)
    check("shuffled vs. sorted-order loader give comparably scaled ratios",
          rel_diff < 0.5,
          f"ratio_shuffled={ratio_shuf:.2f}x ratio_sorted(first-15-only)={ratio_sort:.2f}x "
          f"rel_diff={rel_diff:.2f}")
    print("  NOTE: large divergence here is expected/OK if it's due to which")
    print("  15-of-30 classes were sampled, not due to a code bug. Inspect")
    print("  manually if this fails -- it's a warning sign, not a hard assert.")

    print()
    print("=" * 70)
    print("TEST 5: Edge case -- fewer than 2 usable classes returns default")
    print("=" * 70)
    ds_tiny = SyntheticClusterDataset(num_classes=1, samples_per_class=10,
                                       dim=16, cluster_std=0.1, seed=5)
    loader_tiny = DataLoader(ds_tiny, batch_size=8, shuffle=True, num_workers=0)
    is_fine, acc, chance, ratio = check_fine_grained(
        model, loader_tiny, ratio_threshold=3.0, max_classes=5,
        max_per_class=10, support_ratio=0.5, seed=0, device=device,
    )
    check("single-class input returns safe default (False, 0.0, 0.0, 1.0)",
          (is_fine, acc, chance, ratio) == (False, 0.0, 0.0, 1.0),
          f"got {(is_fine, acc, chance, ratio)}")

    print()
    print("=" * 70)
    print("TEST 6: Seed stability -- multiple seeds on same data, low variance")
    print("=" * 70)
    loader = make_synthetic_loader(
        num_classes=20, samples_per_class=60, dim=64,
        cluster_std=0.4, seed=6,
    )
    ratios_across_seeds = []
    for seed in range(5):
        _, acc, chance, ratio = check_fine_grained(
            model, loader, ratio_threshold=3.0, max_classes=20,
            max_per_class=60, support_ratio=0.5, seed=seed, device=device,
        )
        ratios_across_seeds.append(ratio)
    mean_r = sum(ratios_across_seeds) / len(ratios_across_seeds)
    std_r = (sum((r - mean_r) ** 2 for r in ratios_across_seeds) / len(ratios_across_seeds)) ** 0.5
    check("ratio is reasonably stable across seeds (std < 25% of mean)",
          std_r < 0.25 * mean_r,
          f"ratios={[round(r,2) for r in ratios_across_seeds]} mean={mean_r:.2f} std={std_r:.2f}")

    print()
    print("=" * 70)
    print(f"SYNTHETIC TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


# ---------------------------------------------------------------------------
# Real-data calibration run
# ---------------------------------------------------------------------------
def run_real_calibration(args):
    from check_granularity import build_datasets, build_encoder, build_loader

    cub_root = args.cub_root if args.cub_root else None
    mini_root = args.mini_imagenet_root if args.mini_imagenet_root else None

    print(f"Building datasets under {args.root} "
          f"(CUB: {cub_root or 'skipped'}, miniImageNet: {mini_root or 'skipped'}) ...")
    datasets = build_datasets(args.root, cub_root=cub_root, mini_imagenet_root=mini_root)

    print(f"Building encoder (pretrained={args.pretrained}) ...")
    model = build_encoder(pretrained=args.pretrained, device=args.device)

    n_known, n_correct = 0, 0
    for name, dataset in datasets.items():
        loader = build_loader(dataset, batch_size=args.batch_size,
                               num_workers=args.num_workers)
        is_fine, acc, chance, ratio = check_fine_grained(
            model, loader, ratio_threshold=args.ratio_threshold,
            max_classes=args.max_classes, max_per_class=args.max_per_class,
            support_ratio=args.support_ratio, seed=args.seed, device=args.device,
        )
        pred = "fine" if is_fine else "coarse"
        truth = GROUND_TRUTH.get(name, "unknown")
        agree = (pred == truth) if truth != "unknown" else None
        agree_str = "-" if agree is None else ("YES" if agree else "NO")
        print(f"{name:15s}  acc={acc:.4f}  ratio={ratio:6.2f}x  "
              f"predicted={pred:6s}  truth={truth:6s}  agree={agree_str}")
        if agree is not None:
            n_known += 1
            n_correct += int(agree)

    if n_known > 0:
        print(f"\nCalibration accuracy: {n_correct}/{n_known} "
              f"({100*n_correct/n_known:.1f}%) at ratio_threshold={args.ratio_threshold}")
    return n_correct, n_known


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", default=False)
    parser.add_argument("--real", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--cub-root", type=str, default="")
    parser.add_argument("--mini-imagenet-root", type=str, default="")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--ratio-threshold", type=float, default=3.0)
    parser.add_argument("--max-classes", type=int, default=50)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--support-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.synthetic and not args.real:
        args.synthetic = True

    ok = True
    if args.synthetic:
        ok = run_synthetic_tests(device=args.device) and ok
    if args.real:
        run_real_calibration(args)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()