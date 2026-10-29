#!/usr/bin/env python
"""
dataset_feature_stats.py

PRE-TRAINING (no gradients, no classifier fitting) feature-space statistics,
computed to help JUSTIFY -- not derive -- the rotation-vs-mix augmentation
split you already determined empirically from real FSCIL session accuracy:

    ROTATION (G1 in your latest labeling): mini_imagenet, cifar100,
        fgvcaircraft, dtd, food101
    MIX      (G2 in your latest labeling): cub200, gtsrb, stl10

--------------------------------------------------------------------------
IMPORTANT SCOPE NOTE -- read this before citing any number this prints
--------------------------------------------------------------------------
This script does NOT search over an open-ended space of possible dataset
properties looking for one that happens to reproduce your split. That
would be post-hoc rationalization: with 8 datasets and a binary split
there are 2^8 = 256 possible groupings, and an unconstrained search will
always turn up *some* property that correlates with any one of them by
chance, regardless of whether it means anything.

Instead, THREE candidate global statistics are pre-registered below, each
with a stated mechanism for why it could plausibly predict rotation-vs-mix
preference (see docstrings on each compute_* function). Only those three
are computed and reported. No additional statistics were computed and
discarded for not fitting -- what you see here is everything that was
tried.

With n=8 datasets (5 vs 3), NONE of this reaches conventional statistical
significance no matter what the numbers say, and the exact permutation
test below will say so honestly. Treat this as evidence-informed narrative
framing for why the split is plausible, not as proof of what causes it.
Testing three metrics against one split is also three implicit comparisons;
no multiple-comparison correction is applied, and the p-values should be
read as "descriptive," not "confirmatory."

RESULTS AS OF THE LAST FULL 8-DATASET RUN (for reference -- rerun if the
group assignments or dataset list change):
    interclass_sep:       p = 0.768  -> no reliable signal
    intraclass_compact:   p = 0.679  -> no reliable signal
    eff_dim:               p = 0.125  -> weak trend only, driven heavily by
                            miniImageNet's outlying value (96.1 vs next
                            highest 63.7); NOT significant at alpha=0.05,
                            and this test's floor at n=8 (5-vs-3) is 0.018,
                            so 0.125 is far from the ceiling of what this
                            design could show even in the best case.
None of the three statistics are strong enough on their own to justify a
causal claim. eff_dim is the only one worth mentioning as a hypothesis in
a write-up, and only with the "trend, not proof" caveat attached.
--------------------------------------------------------------------------
SIGMA / T-DRAW SENSITIVITY -- separate from the group-split analysis above
--------------------------------------------------------------------------
This addresses a different question: given a dataset already routed to the
ROTATION branch, how much does the CROWDED-branch pipeline's effect on
feature geometry change with (a) the Gaussian-noise scale sigma, and (b)
which random orthonormal matrix T happens to get drawn?

The augmentation pipeline reconstructed below (pair-blend -> rotate by T ->
permutation mix -> rotate back by T^T -> add sigma-scaled noise) is an
approximation of the paper's corrected Eqs. (5)-(7) for the purpose of
measuring geometric sensitivity. It intentionally does NOT retrain a
classifier or reproduce FSCIL session accuracy -- it only recomputes the
same pre-registered feature statistics (eff_dim, interclass_sep) on the
augmented features. Treat the numbers here as a sensitivity check on
feature geometry, not as a substitute for an accuracy-based sensitivity
sweep; before citing these numbers in the paper, confirm the algebra here
matches your actual training code (check_augmentation.py / the real
training loop), since a hand-reconstructed pipeline can silently diverge
from the implementation.

--------------------------------------------------------------------------
LOCAL (PER-CLASS) DENSITY CHECK -- separate from both sections above
--------------------------------------------------------------------------
This addresses a third, distinct question: the routing statistic eff_dim
is a single GLOBAL number per dataset, pooling all sampled classes before
computing the participation ratio. A global average can in principle mask
a LOCALLY crowded subgroup of classes even when the dataset as a whole
looks separated, or vice versa. This section checks that directly, using
one pre-registered local statistic (a k-NN local participation-ratio
score, see compute_local_neighborhood_density docstring), and reports
whether ANY class's local neighborhood crosses tau in the opposite
direction of the dataset's global routing label. Like the sigma/T-draw
section, this is a feature-geometry screening check, not an accuracy
sensitivity measure, and local neighborhoods (k+1 centroids) are a much
smaller sample than the full pooled covariance, so treat flags as
"worth a closer look," not as a precise per-class score.

Usage:
    python dataset_feature_stats.py --root ./data \\
        --cub-root /home/rb/DA-main/CUB_200_2011 \\
        --mini-imagenet-root /path/to/mini_imagenet \\
        --sigma-grid 0.0,0.05,0.1,0.2,0.4 --num-t-draws 10 \\
        --tau 46.6 --local-density-k 5
"""

import argparse
import itertools
import os
import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Known split (from real FSCIL session-accuracy results, see chat history --
# NOT derived by this script)
# ---------------------------------------------------------------------------
ROTATION_DATASETS = {"CIFAR100", "miniImageNet", "DTD", "Food101", "FGVCAircraft"}
MIX_DATASETS = {"CUB200", "GTSRB", "STL10"}
KNOWN_GROUP = {name: "rotation" for name in ROTATION_DATASETS}
KNOWN_GROUP.update({name: "mix" for name in MIX_DATASETS})


# ---------------------------------------------------------------------------
# Encoder + dataset plumbing (same as check_augmentation.py)
# ---------------------------------------------------------------------------
class EncoderQWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def encoder_q(self, x):
        return self.backbone(x), None


def build_encoder(pretrained: bool, device="cuda"):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = torchvision.models.resnet18(weights=weights)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
    return EncoderQWrapper(backbone).eval().to(device)


class CUB200(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.transform = transform
        images_txt = os.path.join(root, "images.txt")
        labels_txt = os.path.join(root, "image_class_labels.txt")
        split_txt = os.path.join(root, "train_test_split.txt")
        for p in (images_txt, labels_txt, split_txt):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Expected CUB-200-2011 metadata file not found: {p}")
        id_to_path = {}
        with open(images_txt) as f:
            for line in f:
                img_id, rel_path = line.strip().split(maxsplit=1)
                id_to_path[img_id] = rel_path
        id_to_label = {}
        with open(labels_txt) as f:
            for line in f:
                img_id, class_id = line.strip().split()
                id_to_label[img_id] = int(class_id) - 1
        id_to_split = {}
        with open(split_txt) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                id_to_split[img_id] = int(is_train)
        want_train = 1 if split == "train" else 0
        self.samples = []
        for img_id, rel_path in id_to_path.items():
            if id_to_split.get(img_id) == want_train:
                full_path = os.path.join(root, "images", rel_path)
                self.samples.append((full_path, id_to_label[img_id]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No CUB-200-2011 samples found for split='{split}' under {root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class MiniImageNet(Dataset):
    def __init__(self, root, csv_name="train.csv", transform=None):
        import csv
        self.root = root
        self.transform = transform
        csv_path = os.path.join(root, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Expected miniImageNet split file not found: {csv_path}")
        label_to_idx = {}
        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname, label = row["filename"], row["label"]
                if label not in label_to_idx:
                    label_to_idx[label] = len(label_to_idx)
                self.samples.append((os.path.join(root, "images", fname), label_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_datasets(root, cub_root=None, mini_imagenet_root=None):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = {}
    ds["CIFAR100"] = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform)
    ds["STL10"] = torchvision.datasets.STL10(root, split="train", download=True, transform=transform)
    ds["GTSRB"] = torchvision.datasets.GTSRB(root, split="train", download=True, transform=transform)
    if mini_imagenet_root is not None:
        ds["miniImageNet"] = MiniImageNet(mini_imagenet_root, transform=transform)
    ds["FGVCAircraft"] = torchvision.datasets.FGVCAircraft(root, split="train", download=True, transform=transform)
    ds["Food101"] = torchvision.datasets.Food101(root, split="train", download=True, transform=transform)
    ds["DTD"] = torchvision.datasets.DTD(root, split="train", download=True, transform=transform)
    if cub_root is not None:
        ds["CUB200"] = CUB200(cub_root, split="train", transform=transform)
    return ds


def build_loader(dataset, batch_size=64, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def extract_features(model, dataloader, pool_classes, max_per_class, device):
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
            l = l.item()
            if len(feats_per_class[l]) < max_per_class:
                feats_per_class[l].append(f.detach().cpu())
        if len(feats_per_class) >= pool_classes and all(len(v) >= max_per_class for v in feats_per_class.values()):
            break
    return {k: v for k, v in feats_per_class.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# Pre-registered candidate global statistics (see module docstring for
# mechanisms)
# ---------------------------------------------------------------------------
def compute_interclass_separability(feats_per_class):
    """
    Mean pairwise cosine similarity between class centroids (normalized
    mean feature per class). LOWER = classes are further apart in feature
    space = more separable.
    """
    centroids = []
    for label, feats in feats_per_class.items():
        c = torch.stack(feats).mean(0)
        centroids.append(F.normalize(c, dim=0))
    C = torch.stack(centroids)
    sims = C @ C.T
    n = sims.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    return sims[mask].mean().item()


def compute_intraclass_compactness(feats_per_class):
    """
    Mean cosine DISTANCE (1 - cosine similarity) of samples to their own
    class centroid, averaged over classes. LOWER = tighter, more compact
    clusters per class.
    """
    per_class_means = []
    for label, feats in feats_per_class.items():
        F_stack = F.normalize(torch.stack(feats), dim=1)
        centroid = F.normalize(F_stack.mean(0, keepdim=True), dim=1)
        dist = 1.0 - (F_stack @ centroid.T).squeeze(1)
        per_class_means.append(dist.mean().item())
    return statistics.mean(per_class_means)


def compute_effective_dimensionality(feats_per_class, max_samples=2000):
    """
    Participation ratio of the pooled feature covariance matrix:
        PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    Ranges from 1 (all variance in one direction) to D (variance spread
    evenly across all D dimensions). HIGHER = features occupy a larger,
    more isotropic subspace.
    """
    all_feats = []
    for feats in feats_per_class.values():
        all_feats.extend(feats)
    if len(all_feats) > max_samples:
        import random
        all_feats = random.sample(all_feats, max_samples)
    X = torch.stack(all_feats)
    X = X - X.mean(0, keepdim=True)
    cov = (X.T @ X) / (X.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    pr = (eigvals.sum() ** 2) / (eigvals.pow(2).sum() + 1e-12)
    return pr.item()


# ---------------------------------------------------------------------------
# Honest small-sample group comparison: exact permutation test
# ---------------------------------------------------------------------------
def exact_permutation_test(values_by_name, group_by_name):
    """
    Two-sample mean-difference permutation test, EXACT (enumerates every
    way to split these specific n datasets into groups of the same sizes
    as rotation/mix), not approximate. Appropriate given how small n is
    here -- exact enumeration is cheap (C(n, k) grows slowly for these
    sizes) and avoids any Monte-Carlo approximation error.

    Returns (observed_diff, p_two_sided, min_possible_p). min_possible_p is
    1/C(n,k) -- the smallest p this exact test could EVER report for this
    n and this group split, i.e. the resolution ceiling. A p close to
    min_possible_p is meaningfully different from a p that's merely
    "smallish"; a p far from it, even if under 0.1, is weak by comparison.
    This does NOT correct for testing multiple statistics at once (see
    module docstring) -- treat each p as descriptive on its own, and treat
    the whole set as exploratory, not confirmatory.
    """
    names = list(values_by_name.keys())
    rot = [n for n in names if group_by_name[n] == "rotation"]
    mix = [n for n in names if group_by_name[n] == "mix"]
    k = len(rot)  # rotation-group size
    observed = statistics.mean(values_by_name[n] for n in rot) - statistics.mean(values_by_name[n] for n in mix)

    all_vals = [values_by_name[n] for n in names]
    count_as_extreme = 0
    total = 0
    for combo in itertools.combinations(range(len(names)), k):
        combo_set = set(combo)
        a = [all_vals[i] for i in combo_set]
        b = [all_vals[i] for i in range(len(names)) if i not in combo_set]
        diff = statistics.mean(a) - statistics.mean(b)
        if abs(diff) >= abs(observed) - 1e-12:
            count_as_extreme += 1
        total += 1
    p_two_sided = count_as_extreme / total
    min_possible_p = 1.0 / total
    return observed, p_two_sided, min_possible_p


def interpret_p(p, min_possible_p, alpha=0.05, trend_alpha=0.15):
    """
    Turns a p-value + this test's resolution ceiling into a plain-English
    call, so the report doesn't rely on eyeballing numbers. Thresholds are
    conventional (alpha=0.05 for "significant"), plus a looser trend_alpha
    band to flag things worth mentioning as a hypothesis without claiming
    they're proven -- exactly the distinction that matters when n is this
    small and min_possible_p is nowhere near alpha itself.
    """
    if p <= alpha:
        return f"SIGNIFICANT at alpha={alpha} (still small-n evidence -- corroborate on more datasets before relying on it)"
    elif p <= trend_alpha:
        return f"weak trend only (p={p:.3f} vs alpha={alpha}; test's own floor is {min_possible_p:.3f}) -- worth mentioning as a hypothesis, not a result"
    else:
        return f"no reliable signal (p={p:.3f}) -- indistinguishable from a random regrouping of the same numbers"


# ---------------------------------------------------------------------------
# Sigma / T-draw sensitivity (CROWDED / rotation branch only)
# ---------------------------------------------------------------------------
def generate_random_orthonormal_matrix(dim, generator=None):
    """
    Draws a random orthonormal DxD matrix via QR decomposition of a Gaussian
    random matrix, with the sign of each column fixed so that draws are
    reproducible given a generator seed (otherwise QR's sign is arbitrary,
    which would make "different T draws" partly an artifact of QR's
    convention rather than a genuine independent draw).
    """
    A = torch.randn(dim, dim, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R).sign()
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q


def apply_orbit_style_augmentation(feats_per_class, T_matrix, sigma, pair_alpha=0.5, generator=None):
    """
    Approximate reconstruction of the CROWDED-branch pipeline for the
    purpose of a GEOMETRIC sensitivity check only:
        1. pair-blend within class (adopted from FACL, Eq. 4)
        2. rotate by T                                  (Eq. 5)
        3. permutation-based mixing across channels
        4. rotate back by T^T (T is orthonormal, so T^{-1} = T^T)  (Eq. 7)
        5. add Gaussian noise scaled by sigma

    This is NOT guaranteed to be bit-for-bit identical to the actual
    training-time implementation -- verify against the real code before
    citing numbers from this function in the paper. It exists so that
    eff_dim / interclass_sep can be recomputed on augmented features to see
    how much they move as sigma and the T draw change, holding everything
    else (the real extracted base features) fixed.
    """
    if T_matrix.dim() != 2 or T_matrix.size(0) != T_matrix.size(1):
        raise ValueError("T_matrix must be a square DxD orthonormal matrix.")
    dim = T_matrix.size(0)
    permutation = torch.randperm(dim, generator=generator)

    augmented = {}
    for label, feats in feats_per_class.items():
        stacked = torch.stack(feats)
        n = stacked.size(0)
        if stacked.size(1) != dim:
            raise ValueError(f"Feature dim {stacked.size(1)} does not match T_matrix dim {dim}.")
        if n < 2:
            augmented[label] = feats
            continue
        perm_idx = torch.randperm(n, generator=generator)
        blended = pair_alpha * stacked + (1.0 - pair_alpha) * stacked[perm_idx]
        rotated = blended @ T_matrix
        mixed = rotated[:, permutation]
        rotated_back = mixed @ T_matrix.T
        noise = sigma * torch.randn(rotated_back.shape, generator=generator)
        aug = rotated_back + noise
        augmented[label] = [aug[i] for i in range(aug.size(0))]
    return augmented


def summarize_sigma_T_sensitivity(feats_per_class, sigma_grid, num_t_draws, seed_base=0,
                                   stable_cv_threshold=0.05):
    """
    For each sigma in sigma_grid, draws num_t_draws independent random T
    matrices (and independent channel permutations / pairings, since those
    are also part of the "random draw" in the real pipeline), applies the
    augmentation, and recomputes eff_dim and interclass_sep on the result.

    Reports, per sigma: mean +/- population std across T draws, and the
    coefficient of variation (std/mean) as a scale-free stability measure.
    A dataset/statistic pair is called "stable across T draws" only if the
    CV is under stable_cv_threshold (default 5%); otherwise it's flagged as
    draw-sensitive, honestly, rather than papered over with a single
    cherry-picked seed.

    This says nothing about downstream FSCIL accuracy -- only about how
    much the augmentation moves these two feature-geometry statistics.
    """
    dim = next(iter(feats_per_class.values()))[0].shape[0]
    per_sigma = {}
    for sigma in sigma_grid:
        eff_vals, sep_vals = [], []
        for draw in range(num_t_draws):
            g = torch.Generator().manual_seed(seed_base + draw)
            T_matrix = generate_random_orthonormal_matrix(dim, generator=g)
            aug_feats = apply_orbit_style_augmentation(feats_per_class, T_matrix, sigma, generator=g)
            eff_vals.append(compute_effective_dimensionality(aug_feats))
            sep_vals.append(compute_interclass_separability(aug_feats))

        def _cv(vals):
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cv = (s / abs(m)) if m != 0 else float("inf")
            return m, s, cv

        eff_mean, eff_std, eff_cv = _cv(eff_vals)
        sep_mean, sep_std, sep_cv = _cv(sep_vals)
        per_sigma[sigma] = {
            "eff_dim_mean": eff_mean, "eff_dim_std": eff_std, "eff_dim_cv": eff_cv,
            "interclass_sep_mean": sep_mean, "interclass_sep_std": sep_std, "interclass_sep_cv": sep_cv,
            "num_draws": num_t_draws,
        }
    return per_sigma, stable_cv_threshold


def print_sigma_T_sensitivity_report(dataset_name, per_sigma, stable_cv_threshold):
    print(f"\n--- {dataset_name} (sigma / T-draw sensitivity, {next(iter(per_sigma.values()))['num_draws']} T draws per sigma) ---")
    header = f"{'sigma':>6s} {'eff_dim mean+-std':>20s} {'CV':>7s} {'sep mean+-std':>18s} {'CV':>7s}  verdict"
    print(header)
    print("-" * len(header))
    for sigma in sorted(per_sigma.keys()):
        r = per_sigma[sigma]
        eff_str = f"{r['eff_dim_mean']:.2f}+-{r['eff_dim_std']:.2f}"
        sep_str = f"{r['interclass_sep_mean']:.4f}+-{r['interclass_sep_std']:.4f}"
        stable = r["eff_dim_cv"] <= stable_cv_threshold and r["interclass_sep_cv"] <= stable_cv_threshold
        verdict = "stable across T draws" if stable else "draw-sensitive (CV above threshold)"
        print(f"{sigma:6.2f} {eff_str:>20s} {r['eff_dim_cv']:7.3f} {sep_str:>18s} {r['interclass_sep_cv']:7.3f}  {verdict}")

    eff_means_by_sigma = [per_sigma[s]["eff_dim_mean"] for s in sorted(per_sigma.keys())]
    monotonic_up = all(b >= a - 1e-9 for a, b in zip(eff_means_by_sigma, eff_means_by_sigma[1:]))
    monotonic_down = all(b <= a + 1e-9 for a, b in zip(eff_means_by_sigma, eff_means_by_sigma[1:]))
    if monotonic_up and not monotonic_down:
        trend = "eff_dim increases monotonically with sigma across the tested grid."
    elif monotonic_down and not monotonic_up:
        trend = "eff_dim decreases monotonically with sigma across the tested grid."
    else:
        trend = "eff_dim does not move monotonically with sigma across the tested grid -- no simple trend to report."
    print(f"    sigma trend: {trend}")
    print("    Caveat: this is a feature-geometry proxy (no classifier retrained); "
          "confirm against real FSCIL session accuracy before citing as accuracy sensitivity.")


# ---------------------------------------------------------------------------
# Local (per-class) crowding density check
# ---------------------------------------------------------------------------
# MOTIVATION: the routing statistic eff_dim (Eq. crowding in the paper) is a
# single GLOBAL number per dataset -- it pools all sampled classes together
# before computing the participation ratio. A reasonable concern (raised in
# review) is that a global average can mask a LOCALLY crowded subgroup of
# classes even when the dataset as a whole looks separated, or vice versa.
#
# This section checks that concern directly, using a single pre-registered
# local statistic (not a search over many candidate local statistics -- see
# the same post-hoc-rationalization caveat in the module docstring above).
#
# METHOD: for each class centroid, we look at its k nearest neighboring
# class centroids (by cosine similarity) and compute a local participation-
# ratio-style density score over that class's centroid plus its k neighbors.
# This gives a LOCAL analogue of the global eff_dim statistic, restricted to
# each class's most crowded local neighborhood rather than averaged over the
# whole dataset. The functional form matches compute_effective_dimensionality
# exactly, so local and global scores are on a comparable scale and can be
# checked against the same threshold tau.
#
# WHAT COUNTS AS A FLAG: a dataset's GLOBAL regime (Crowded/Separated) is
# already fixed by its global eff_dim vs. tau. We flag a dataset if ANY
# class's local neighborhood density crosses tau in the OPPOSITE direction
# of the global label -- e.g., a globally Separated dataset that contains a
# locally Crowded pocket of classes, or vice versa. This is the specific
# failure mode raised in review (a global average masking local structure),
# and is what this check is designed to catch, not to prove structure is
# ALWAYS uniform.
#
# LIMITATION, stated up front: k-NN neighborhoods of size k+1 are much
# smaller samples than the full pooled covariance used for the global
# statistic, so local density estimates are noisier and should be read as a
# screening tool (does ANY neighborhood cross the line) rather than a
# precise per-class score. This does not retrain a classifier and is not a
# per-class accuracy sensitivity measure.
# ---------------------------------------------------------------------------
def compute_local_neighborhood_density(feats_per_class, k=5):
    """
    For each class, finds its k nearest other classes by cosine similarity
    between centroids, then computes a local participation-ratio-style
    density score over that class's centroid plus its k neighbors.

    Returns: dict {class_label: local_density_score}

    local_density_score uses the same functional form as eff_dim
    (participation ratio of the local centroid covariance), so a HIGHER
    score again means a more isotropic / crowded local neighborhood,
    consistent with the sign convention used by the global eff_dim
    statistic and the paper's routing rule (Eq. crowding, Eq. regime).
    """
    labels = list(feats_per_class.keys())
    centroids = {}
    for label in labels:
        c = torch.stack(feats_per_class[label]).mean(0)
        centroids[label] = F.normalize(c, dim=0)

    if len(labels) < k + 2:
        raise ValueError(
            f"Need at least {k + 2} classes to form {k}-NN neighborhoods; "
            f"got {len(labels)}. Reduce --local-density-k or increase --pool-classes."
        )

    C = torch.stack([centroids[l] for l in labels])  # [n_classes, D]
    sims = C @ C.T  # cosine similarity, since centroids are normalized
    local_density = {}
    for i, label in enumerate(labels):
        row = sims[i].clone()
        row[i] = -float("inf")  # exclude self
        topk_idx = torch.topk(row, k).indices
        neighborhood_idx = torch.cat([torch.tensor([i]), topk_idx])
        local_centroids = C[neighborhood_idx]  # [k+1, D]

        # participation ratio of the local centroid "covariance", same form
        # as compute_effective_dimensionality, but over just this neighborhood
        X = local_centroids - local_centroids.mean(0, keepdim=True)
        cov = (X.T @ X) / max(1, X.size(0) - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
        pr = (eigvals.sum() ** 2) / (eigvals.pow(2).sum() + 1e-12)
        local_density[label] = pr.item()

    return local_density


def check_local_density_against_global_regime(dataset_name, local_density_by_class,
                                                 global_eff_dim, tau, k):
    """
    Compares each class's local density score against tau, and reports
    whether any class's local neighborhood crosses tau in the OPPOSITE
    direction of the dataset's global routing label. This is the concrete
    check for "does a global average mask a locally crowded/separated
    subgroup".

    Returns: (global_regime_str, n_flagged, flagged_classes, summary_str)
    """
    global_regime = "Crowded" if global_eff_dim >= tau else "Separated"
    flagged = []
    for label, score in local_density_by_class.items():
        local_regime = "Crowded" if score >= tau else "Separated"
        if local_regime != global_regime:
            flagged.append((label, score, local_regime))

    n_total = len(local_density_by_class)
    n_flagged = len(flagged)
    if n_flagged == 0:
        summary = (
            f"{dataset_name}: global={global_regime} (eff_dim={global_eff_dim:.2f}, "
            f"tau={tau:.1f}). All {n_total} class-local {k}-NN neighborhoods agree "
            f"with the global regime -- no evidence of a locally crowded/separated "
            f"pocket at this k."
        )
    else:
        summary = (
            f"{dataset_name}: global={global_regime} (eff_dim={global_eff_dim:.2f}, "
            f"tau={tau:.1f}). {n_flagged}/{n_total} class-local {k}-NN neighborhoods "
            f"cross tau in the OPPOSITE direction (local regime differs from global) "
            f"-- WORTH FLAGGING, see per-class values below. This does not "
            f"automatically mean the global routing decision is wrong for the whole "
            f"dataset, only that local structure is not perfectly uniform; treat as "
            f"a caveat on the global statistic, not a refutation of it."
        )
    return global_regime, n_flagged, flagged, summary


def print_local_density_report(dataset_name, local_density_by_class, global_eff_dim, tau, k):
    global_regime, n_flagged, flagged, summary = check_local_density_against_global_regime(
        dataset_name, local_density_by_class, global_eff_dim, tau, k)

    print(f"\n--- {dataset_name} (local {k}-NN density check) ---")
    print(summary)
    if n_flagged > 0:
        print(f"    Flagged classes (local regime differs from global {global_regime}):")
        for label, score, local_regime in sorted(flagged, key=lambda x: x[1], reverse=True):
            print(f"      class={label!r}  local_density={score:.2f}  local_regime={local_regime}")
    mean_local = statistics.mean(local_density_by_class.values())
    std_local = statistics.pstdev(local_density_by_class.values())
    print(f"    Local density across all classes: mean={mean_local:.2f}  std={std_local:.2f}  "
          f"(for reference, global eff_dim={global_eff_dim:.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--cub-root", type=str, default="/home/rb/DA-main/CUB_200_2011",
                         help="Path to CUB_200_2011 folder. Pass '' to skip.")
    parser.add_argument("--mini-imagenet-root", type=str, default="",
                         help="Path to miniImageNet folder. Pass '' to skip.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pool-classes", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sigma-grid", type=str, default="0.0,0.05,0.1,0.2,0.4",
                         help="Comma-separated sigma (noise std) values to sweep for the "
                              "sigma/T-draw sensitivity analysis on the rotation-branch datasets.")
    parser.add_argument("--num-t-draws", type=int, default=10,
                         help="Number of independent random T draws per sigma value.")
    parser.add_argument("--skip-sigma-t-sensitivity", action="store_true", default=False,
                         help="Skip the sigma/T-draw sensitivity section entirely.")
    parser.add_argument("--tau", type=float, default=46.6,
                         help="Routing threshold tau (must match the value used to "
                              "compute the paper's global Crowded/Separated labels).")
    parser.add_argument("--local-density-k", type=int, default=5,
                         help="Number of nearest neighbor classes per local density "
                              "neighborhood in the local crowding check.")
    parser.add_argument("--skip-local-density-check", action="store_true", default=False,
                         help="Skip the local per-class k-NN density check entirely.")
    args = parser.parse_args()

    cub_root = args.cub_root if args.cub_root else None
    mini_root = args.mini_imagenet_root if args.mini_imagenet_root else None
    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip() != ""]

    print(f"Building datasets under {args.root} "
          f"(CUB: {cub_root or 'skipped'}, miniImageNet: {mini_root or 'skipped'}) ...")
    datasets = build_datasets(args.root, cub_root=cub_root, mini_imagenet_root=mini_root)

    print(f"Building encoder (pretrained={args.pretrained}) ...")
    model = build_encoder(pretrained=args.pretrained, device=args.device)

    interclass_sep = {}
    intraclass_compact = {}
    eff_dim = {}
    n_classes = {}
    all_feats_per_class = {}  # kept around for the sigma/T and local-density sections below

    for name, dataset in datasets.items():
        print(f"\n--- {name} ---")
        loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        feats_per_class = extract_features(model, loader, args.pool_classes, args.max_per_class, args.device)
        if len(feats_per_class) < 2:
            print(f"{name}: too few usable classes ({len(feats_per_class)}), skipping.")
            continue

        sep = compute_interclass_separability(feats_per_class)
        compact = compute_intraclass_compactness(feats_per_class)
        pr = compute_effective_dimensionality(feats_per_class)

        interclass_sep[name] = sep
        intraclass_compact[name] = compact
        eff_dim[name] = pr
        n_classes[name] = len(feats_per_class)
        all_feats_per_class[name] = feats_per_class

        print(f"{name:15s}  interclass_sep={sep:.4f}  intraclass_compact={compact:.4f}  "
              f"eff_dim={pr:.2f}  (n_classes_pool={len(feats_per_class)})")

    # --- Report table ---
    print("\n=== Pre-training feature-space statistics vs. known augmentation group ===")
    header = f"{'Dataset':15s} {'group':>9s} {'interclass_sep':>15s} {'intraclass_compact':>19s} {'eff_dim':>9s}"
    print(header)
    print("-" * len(header))
    for name in sorted(interclass_sep.keys(), key=lambda n: (KNOWN_GROUP.get(n, "?"), n)):
        grp = KNOWN_GROUP.get(name, "?")
        print(f"{name:15s} {grp:>9s} {interclass_sep[name]:15.4f} {intraclass_compact[name]:19.4f} {eff_dim[name]:9.2f}")

    # --- Exact permutation tests, one per pre-registered global statistic ---
    n_total = len(interclass_sep)
    n_rot = sum(1 for n in interclass_sep if KNOWN_GROUP.get(n) == "rotation")
    n_mix = sum(1 for n in interclass_sep if KNOWN_GROUP.get(n) == "mix")

    print("\n=== Exact permutation test: rotation-group mean vs mix-group mean ===")
    print("Read these as descriptive, not confirmatory -- three statistics tested, no multiple-comparison correction.\n")

    findings = []  # (stat_name, verdict_str, p) for the final summary
    for stat_name, values in [
        ("interclass_sep (lower = more separable)", interclass_sep),
        ("intraclass_compact (lower = tighter clusters)", intraclass_compact),
        ("eff_dim (higher = more isotropic feature space)", eff_dim),
    ]:
        if len(values) < 4 or n_rot < 2 or n_mix < 2:
            print(f"{stat_name}: skipped (need at least 2 datasets in each group; "
                  f"currently have {n_rot} rotation / {n_mix} mix)")
            continue
        observed, p, min_p = exact_permutation_test(values, KNOWN_GROUP)
        rot_mean = statistics.mean(v for n, v in values.items() if KNOWN_GROUP[n] == "rotation")
        mix_mean = statistics.mean(v for n, v in values.items() if KNOWN_GROUP[n] == "mix")
        verdict = interpret_p(p, min_p)
        print(f"{stat_name}")
        print(f"    rotation-group mean = {rot_mean:.4f}   mix-group mean = {mix_mean:.4f}   "
              f"diff = {observed:+.4f}")
        print(f"    exact two-sided p = {p:.4f}  (this test's floor at n={n_total}, {n_rot}-vs-{n_mix}: "
              f"{min_p:.4f})")
        print(f"    -> {verdict}\n")
        findings.append((stat_name, verdict, p))

    # --- Final plain-language summary, so nothing requires eyeballing numbers ---
    print("=== Summary ===")
    signal_stats = [f for f in findings if "no reliable signal" not in f[1]]
    if not signal_stats:
        print("None of the three pre-registered statistics showed even a weak trend -- "
              "at this sample size, none of interclass separability, intraclass "
              "compactness, or effective dimensionality distinguish the rotation and "
              "mix groups from a random regrouping of the same numbers. That is a "
              "real (negative) finding: don't build a narrative around any of them.")
    else:
        for stat_name, verdict, p in signal_stats:
            print(f"- {stat_name}: {verdict}")
        print("Everything else tested came back with no reliable signal (see above) "
              "and should not be cited as part of the explanation.")

    print(f"\nCurrent power ceiling: with {n_total} datasets ({n_rot} rotation / {n_mix} mix), "
          f"the smallest p this exact test can EVER report is 1/C({n_total},{n_rot}) = "
          f"{1.0/max(1, len(list(itertools.combinations(range(n_total), n_rot)))):.4f}. "
          f"No amount of re-running will improve this -- the only way to get a test with "
          f"real statistical power is to add more datasets (real FSCIL results, both "
          f"rotation-preferring and mix-preferring) and re-run. Roughly 6+ per group "
          f"is where conventional significance (p<0.05) becomes reachable at all.")

    # --- Sigma / T-draw sensitivity, rotation-branch datasets only ---
    if args.skip_sigma_t_sensitivity:
        print("\n(Sigma/T-draw sensitivity section skipped via --skip-sigma-t-sensitivity.)")
    else:
        rotation_ready = [n for n in all_feats_per_class if KNOWN_GROUP.get(n) == "rotation"]
        print("\n=== Sigma / T-draw sensitivity (CROWDED/rotation branch only) ===")
        print(f"sigma grid: {sigma_grid}   T draws per sigma: {args.num_t_draws}")
        print("This checks how much the augmentation's effect on eff_dim / interclass_sep moves "
              "with sigma and with which random T gets drawn, on the real extracted base features. "
              "It is a feature-geometry proxy, not an accuracy sensitivity sweep -- see module "
              "docstring before citing these numbers.")
        if not rotation_ready:
            print("No rotation-branch datasets had usable extracted features this run -- nothing to report.")
        else:
            for name in sorted(rotation_ready):
                per_sigma, stable_cv_threshold = summarize_sigma_T_sensitivity(
                    all_feats_per_class[name], sigma_grid, args.num_t_draws)
                print_sigma_T_sensitivity_report(name, per_sigma, stable_cv_threshold)

    # --- Local per-class density check, all datasets with usable features ---
    if args.skip_local_density_check:
        print("\n(Local density check skipped via --skip-local-density-check.)")
    else:
        print("\n=== Local per-class k-NN density check (all datasets) ===")
        print(f"k={args.local_density_k}, tau={args.tau}")
        print("Checks whether any class's local neighborhood crosses tau in the "
              "opposite direction of its dataset's global routing label. See "
              "docstring above compute_local_neighborhood_density for method and "
              "limitations before citing these numbers.\n")
        if not all_feats_per_class:
            print("No datasets had usable extracted features this run -- nothing to report.")
        else:
            for name in sorted(all_feats_per_class.keys()):
                try:
                    local_density = compute_local_neighborhood_density(
                        all_feats_per_class[name], k=args.local_density_k)
                    print_local_density_report(
                        name, local_density, eff_dim[name], args.tau, args.local_density_k)
                except ValueError as e:
                    print(f"\n--- {name} (local density check) ---")
                    print(f"    Skipped: {e}")


if __name__ == "__main__":
    main()