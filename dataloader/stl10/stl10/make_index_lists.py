#!/usr/bin/env python
"""
make_index_lists.py

Generates the STL10 FSCIL index-list files this repo's data_utils.py
expects at:
    data/index_list/stl10/session_1.txt   (base classes -- generated for
                                            completeness/inspection; the
                                            base session actually selects
                                            by class id, not this file)
    data/index_list/stl10/session_2.txt   (session 1 novel: horse, monkey -- 5-shot each)
    data/index_list/stl10/session_3.txt   (session 2 novel: ship, truck  -- 5-shot each)

Each line is a single integer: the sample's position in the STL10 TRAIN
split (torchvision's index order), not a file path -- STL10 has no
per-image files. This mirrors this repo's CIFAR100 index-list convention
(which also indexes into a torchvision-loaded array), adapted for STL10.

The split itself (which classes are base/novel, session composition,
shots-per-class) is identical to stl10_fscil.py -- see that script's
module docstring for the full rationale on why 6 base / 2 sessions of
2-way 5-shot was chosen for a 10-class dataset. Kept as literal constants
here (not imported) so this file has no dependency on stl10_fscil.py
being present in the training repo.

Usage:
    python -m dataloader.stl10.make_index_lists --root ./data \\
        --out-dir ./data/index_list/stl10
"""

import argparse
import os
import random

import torchvision

BASE_CLASSES = [0, 1, 2, 3, 4, 5]        # airplane, bird, car, cat, deer, dog
NOVEL_SESSIONS = [[6, 7], [8, 9]]        # session1: horse,monkey ; session2: ship,truck
SHOTS_PER_CLASS = 5
SEED = 0


def _labels_by_class(dataset):
    by_class = {}
    for i, label in enumerate(dataset.labels):
        by_class.setdefault(int(label), []).append(i)
    return by_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./data/index_list/stl10")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = torchvision.datasets.STL10(args.root, split="train", download=True)
    by_class = _labels_by_class(train_ds)
    rng = random.Random(args.seed)

    # session_1.txt: all base-class training samples (not used by
    # get_base_dataloader in the standard convention, but written for
    # inspection/consistency with the other datasets' index_list layout).
    base_indices = []
    for c in BASE_CLASSES:
        base_indices.extend(by_class[c])
    with open(os.path.join(args.out_dir, "session_1.txt"), "w") as f:
        for idx in base_indices:
            f.write(f"{idx}\n")
    print(f"session_1.txt: {len(base_indices)} samples "
          f"({len(BASE_CLASSES)} base classes, full data)")

    # session_2.txt, session_3.txt, ...: K-shot per novel class, sampled
    # deterministically so re-running this script reproduces the same split.
    for s_num, session_classes in enumerate(NOVEL_SESSIONS, start=1):
        session_indices = []
        for c in session_classes:
            pool = by_class[c].copy()
            rng.shuffle(pool)
            if len(pool) < SHOTS_PER_CLASS:
                raise RuntimeError(
                    f"class id {c} has only {len(pool)} training images, "
                    f"fewer than SHOTS_PER_CLASS={SHOTS_PER_CLASS}."
                )
            session_indices.extend(pool[:SHOTS_PER_CLASS])

        fname = os.path.join(args.out_dir, f"session_{s_num + 1}.txt")
        with open(fname, "w") as f:
            for idx in session_indices:
                f.write(f"{idx}\n")
        print(f"session_{s_num + 1}.txt: {len(session_indices)} samples "
              f"({len(session_classes)}-way, {SHOTS_PER_CLASS}-shot: classes {session_classes})")


if __name__ == "__main__":
    main()