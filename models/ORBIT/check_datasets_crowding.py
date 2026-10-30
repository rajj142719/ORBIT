"""
Anisotropy-adjusted crowding metric for dataset regime routing (ORBIT-style).

Motivation
----------
ORBIT's Eq. (2) computes rho, the mean pairwise cosine similarity between
class prototypes, and Eq. (3) routes to CROWDED (rho >= tau) or SEPARATED
(rho < tau) using a single fixed threshold tau=0.70. That tau was tuned on
exactly two anchor points (CIFAR-100/miniImageNet vs. CUB-200), so applying
it unmodified to new, unrelated datasets is an out-of-sample generalization
test, not a re-validated threshold.

A known confound with raw rho is embedding anisotropy: cosine similarities
from a pretrained encoder are inflated by a dataset-and-encoder-dependent
"background" similarity level, even between semantically unrelated points
(Mu & Viswanath, 2018; Ethayarajh, 2019). This script computes an
anisotropy-adjusted crowding score:

    rho_class = mean cosine similarity between class prototypes  (Eq. 2)
    rho_null  = mean cosine similarity between RANDOM sample pairs
                (not grouped by class) from the SAME dataset, SAME encoder
    rho_adj   = (rho_class - rho_null) / (1 - rho_null)

rho_adj asks "are class prototypes closer together than random points from
this same dataset would be by default?" -- isolating true class-crowding
from the encoder's baseline anisotropy, without introducing an external
reference point (e.g. distance-to-ImageNet) that measures domain shift
rather than class separability.

Validation-before-use
----------------------
Before trusting rho_adj on new datasets, this script first computes it on
the three datasets with a known, paper-established ground truth regime:
CIFAR-100 and miniImageNet (CROWDED / coarse-grained) and CUB-200
(SEPARATED / fine-grained). If rho_adj reproduces those regimes under some
threshold, that is evidence the metric is measuring the right thing. If it
does not, that must be reported, not routed around.

CUB-200 and miniImageNet are not in torchvision; point --cub200_root and
--mini_imagenet_root at local ImageFolder-style directories
(class_name/*.jpg) if you have them, or leave unset to skip validation.

Usage:
    python check_datasets_crowding.py --root ./data \
        --cub200_root /path/to/CUB_200_2011/images \
        --mini_imagenet_root /path/to/miniImageNet/images
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Custom dataset for flat-file, CSV-labeled layouts (e.g. Ravi & Larochelle
# miniImageNet splits: images/*.jpg + train.csv with "filename,label" rows)
# ---------------------------------------------------------------------------

class CSVLabeledImageDataset(Dataset):
    """
    Reads a flat image directory plus a CSV of (filename, label) pairs, as
    used by the standard miniImageNet (Ravi & Larochelle) splits. Skips
    macOS resource-fork files (leading "._") automatically. String labels
    are mapped to contiguous integer indices.
    """
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        label_set = set()

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header row (e.g. "filename,label")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                fname, label = row[0].strip(), row[1].strip()
                if fname.startswith("._"):
                    continue
                label_set.add(label)
                self.samples.append((fname, label))

        self.classes = sorted(label_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[label]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_new_datasets(root):
    """
    The 7 out-of-sample datasets being evaluated for generalization of the
    routing rule. All auto-download via torchvision.
    """
    ds = {}
    ds["Flowers102"] = torchvision.datasets.Flowers102(root, split="train", download=True, transform=TRANSFORM)
    ds["OxfordIIITPet"] = torchvision.datasets.OxfordIIITPet(root, split="trainval", download=True, transform=TRANSFORM)
    ds["FGVCAircraft"] = torchvision.datasets.FGVCAircraft(root, split="train", download=True, transform=TRANSFORM)
    ds["Food101"] = torchvision.datasets.Food101(root, split="train", download=True, transform=TRANSFORM)
    ds["DTD"] = torchvision.datasets.DTD(root, split="train", download=True, transform=TRANSFORM)
    ds["STL10"] = torchvision.datasets.STL10(root, split="train", download=True, transform=TRANSFORM)
    # SVHN intentionally excluded: digit crops carry little semantic content
    # for an ImageNet-pretrained backbone, producing an outlier similarity
    # (~0.99) that reflects the encoder's blind spot, not true class
    # crowding. Re-enable only with an encoder actually suited to digits.
    ds["GTSRB"] = torchvision.datasets.GTSRB(root, split="train", download=True, transform=TRANSFORM)
    return ds


def build_validation_datasets(root, cifar100=True, cub200_root=None,
                               mini_imagenet_root=None, mini_imagenet_csv=None):
    """
    The 3 datasets with a known, paper-established ground-truth regime,
    used to sanity-check rho_adj before trusting it on new data.

    mini_imagenet_root: directory containing the flat *.jpg files
        (e.g. .../miniimagenet/images/)
    mini_imagenet_csv: CSV of (filename,label) rows for that split
        (e.g. .../miniimagenet/train.csv) -- required alongside
        mini_imagenet_root for the Ravi & Larochelle flat-file layout.
    """
    ds = {}
    if cifar100:
        ds["CIFAR-100"] = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=TRANSFORM)
    if mini_imagenet_root is not None:
        if mini_imagenet_csv is not None:
            ds["miniImageNet"] = CSVLabeledImageDataset(
                mini_imagenet_root, mini_imagenet_csv, transform=TRANSFORM,
            )
        else:
            # fall back to ImageFolder layout (subfolder-per-class)
            ds["miniImageNet"] = torchvision.datasets.ImageFolder(mini_imagenet_root, transform=TRANSFORM)
    if cub200_root is not None:
        ds["CUB-200"] = torchvision.datasets.ImageFolder(cub200_root, transform=TRANSFORM)
    return ds


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, dataset, device, max_classes=50, max_per_class=20,
                      max_random_samples=2000, batch_size=64, seed=0):
    """
    Single pass over `dataset` that collects:
      - feats_per_class: up to max_per_class embeddings per class, for up
        to max_classes classes (used for rho_class / prototypes)
      - random_feats: up to max_random_samples embeddings drawn without
        regard to class (used for rho_null)
    Both come from the same forward passes, so this costs one pass over the
    data rather than two.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, generator=g)

    feats_per_class = defaultdict(list)
    random_feats = []

    for data, labels in loader:
        data = data.to(device)
        feats, _ = model.encoder_q(data)
        feats = F.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
        feats_cpu = feats.detach().cpu()

        for f, l in zip(feats_cpu, labels):
            l = int(l)
            if l not in feats_per_class and len(feats_per_class) >= max_classes:
                pass
            elif len(feats_per_class[l]) < max_per_class:
                feats_per_class[l].append(f)
            if len(random_feats) < max_random_samples:
                random_feats.append(f)

        enough_classes = len(feats_per_class) >= max_classes and \
            all(len(v) >= max_per_class for v in feats_per_class.values())
        enough_random = len(random_feats) >= max_random_samples
        if enough_classes and enough_random:
            break

    return feats_per_class, random_feats


def compute_rho_class(feats_per_class):
    """Eq. (2): mean pairwise cosine similarity between class prototypes."""
    if len(feats_per_class) < 2:
        return None
    prototypes = [torch.stack(f).mean(0) for f in feats_per_class.values()]
    P = F.normalize(torch.stack(prototypes), dim=1)
    sim = P @ P.T
    mask = ~torch.eye(len(P), dtype=torch.bool)
    return sim[mask].mean().item()


def compute_rho_null(random_feats, num_pairs=20000, seed=0):
    """
    Mean cosine similarity between RANDOM sample pairs (not grouped by
    class) from the same dataset/encoder -- the anisotropy baseline.
    """
    if len(random_feats) < 2:
        return None
    rnd = random.Random(seed)
    X = F.normalize(torch.stack(random_feats), dim=1)
    n = X.size(0)
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    sims = []
    seen = set()
    while len(sims) < num_pairs:
        i, j = rnd.randrange(n), rnd.randrange(n)
        if i == j or (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        sims.append((X[i] @ X[j]).item())
    return sum(sims) / len(sims)


def compute_rho_adj(rho_class, rho_null):
    """rho_adj = (rho_class - rho_null) / (1 - rho_null)."""
    if rho_class is None or rho_null is None or rho_null >= 1.0:
        return None
    return (rho_class - rho_null) / (1.0 - rho_null)


def analyze_dataset(model, dataset, device, max_classes=50, max_per_class=20,
                     max_random_samples=2000, batch_size=64, seed=0):
    feats_per_class, random_feats = extract_features(
        model, dataset, device, max_classes=max_classes,
        max_per_class=max_per_class, max_random_samples=max_random_samples,
        batch_size=batch_size, seed=seed,
    )
    rho_class = compute_rho_class(feats_per_class)
    rho_null = compute_rho_null(random_feats, seed=seed)
    rho_adj = compute_rho_adj(rho_class, rho_null)
    return {
        "rho_class": rho_class,
        "rho_null": rho_null,
        "rho_adj": rho_adj,
        "num_classes": len(feats_per_class),
    }


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(device):
    """
    Generic ImageNet-pretrained backbone wrapped with an encoder_q-style
    interface so analyze_dataset() matches the original single-dataset
    checker's calling convention. Swap in your own trained model when
    available -- everything downstream is unchanged.
    """
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2])  # keep spatial feature map

    class EncoderQWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def encoder_q(self, x):
            feats = self.backbone(x)
            return feats, None

    return EncoderQWrapper(backbone).eval().to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    parser.add_argument("--cub200_root", default=None,
                         help="Path to CUB_200_2011/images (ImageFolder layout). Skips CUB-200 validation if unset.")
    parser.add_argument("--mini_imagenet_root", default=None,
                         help="Path to miniImageNet flat images dir (e.g. .../miniimagenet/images/). "
                              "Skips miniImageNet validation if unset.")
    parser.add_argument("--mini_imagenet_csv", default=None,
                         help="Path to miniImageNet label CSV (e.g. .../miniimagenet/train.csv). "
                              "Required alongside --mini_imagenet_root for the flat-file layout; "
                              "if omitted, falls back to assuming --mini_imagenet_root is an "
                              "ImageFolder (subfolder-per-class) layout instead.")
    parser.add_argument("--skip_cifar100", action="store_true")
    parser.add_argument("--max_classes", type=int, default=50)
    parser.add_argument("--max_per_class", type=int, default=20)
    parser.add_argument("--max_random_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = build_backbone(device)

    # ---- Step 1: validate rho_adj against known-ground-truth datasets ----
    val_datasets = build_validation_datasets(
        args.root,
        cifar100=not args.skip_cifar100,
        cub200_root=args.cub200_root,
        mini_imagenet_root=args.mini_imagenet_root,
        mini_imagenet_csv=args.mini_imagenet_csv,
    )

    print("\n===== VALIDATION: known ground-truth regimes (paper Sec. 3.2 / Sec. 4.1) =====")
    known_regime = {
        "CIFAR-100": "CROWDED (coarse-grained)",
        "miniImageNet": "CROWDED (coarse-grained)",
        "CUB-200": "SEPARATED (fine-grained)",
    }
    val_results = {}
    if not val_datasets:
        print("No validation datasets available (set --cub200_root / --mini_imagenet_root "
              "to run this check). Proceeding to new-dataset analysis without validation "
              "-- results below should be reported as unvalidated if this check is skipped.")
    for name, dataset in val_datasets.items():
        print(f"\n--- {name} (expected: {known_regime[name]}) ---")
        try:
            r = analyze_dataset(
                model, dataset, device,
                max_classes=args.max_classes, max_per_class=args.max_per_class,
                max_random_samples=args.max_random_samples,
                batch_size=args.batch_size, seed=args.seed,
            )
            print(f"{name}: rho_class={r['rho_class']:.4f}  rho_null={r['rho_null']:.4f}  "
                  f"rho_adj={r['rho_adj']:.4f}  (n_classes={r['num_classes']})")
            val_results[name] = r
        except Exception as e:
            print(f"{name}: FAILED ({e})")

    if len(val_results) >= 2:
        print("\nCheck: does rho_adj separate the CROWDED group from CUB-200 with a clear gap?")
        crowded_vals = [v["rho_adj"] for k, v in val_results.items() if k in ("CIFAR-100", "miniImageNet")]
        separated_vals = [v["rho_adj"] for k, v in val_results.items() if k == "CUB-200"]
        if crowded_vals and separated_vals:
            gap = min(crowded_vals) - max(separated_vals)
            print(f"min(CROWDED rho_adj) - max(SEPARATED rho_adj) = {gap:.4f} "
                  f"({'reproduces known regimes' if gap > 0 else 'DOES NOT reproduce known regimes -- do not trust rho_adj on new data without investigating this'})")

    # ---- Step 2: apply the same metric to the new, out-of-sample datasets ----
    new_datasets = build_new_datasets(args.root)

    print("\n===== NEW DATASETS: out-of-sample application of rho_adj =====")
    print("(tau for these has NOT been separately tuned; report accordingly)")
    new_results = []
    for name, dataset in new_datasets.items():
        print(f"\n--- {name} ---")
        try:
            r = analyze_dataset(
                model, dataset, device,
                max_classes=args.max_classes, max_per_class=args.max_per_class,
                max_random_samples=args.max_random_samples,
                batch_size=args.batch_size, seed=args.seed,
            )
            print(f"{name}: rho_class={r['rho_class']:.4f}  rho_null={r['rho_null']:.4f}  rho_adj={r['rho_adj']:.4f}")
            new_results.append((name, r))
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            new_results.append((name, None))

    print("\n===== summary: new datasets sorted by rho_adj =====")
    ok = [(n, r) for n, r in new_results if r is not None]
    ok.sort(key=lambda x: x[1]["rho_adj"])
    for name, r in ok:
        print(f"{name:15s} rho_class={r['rho_class']:.4f}  rho_null={r['rho_null']:.4f}  rho_adj={r['rho_adj']:.4f}")
    print("\nNote: this table reports rho_class, rho_null, and rho_adj for each dataset. "
          "Assigning CROWDED/SEPARATED labels requires a threshold on rho_adj; unlike the "
          "original tau=0.70 (tuned on raw rho for CIFAR-100/miniImageNet/CUB-200 only), no "
          "such threshold has been established for rho_adj here. If the validation step above "
          "reproduced the known regimes, the corresponding rho_adj values for CIFAR-100/"
          "miniImageNet and CUB-200 can be used as reference anchors for where a rho_adj "
          "threshold should fall; report that derivation explicitly rather than picking a "
          "round number.")


if __name__ == "__main__":
    sys.exit(main())