#!/usr/bin/env python
"""
make_index_lists.py (Food101)

Generates FSCIL index-list files at:
    <dataroot>/food101/index_list/session_2.txt .. session_9.txt

Split (101 classes total): base 0-60, novel sessions 61-65,...,96-100
(8 x 5-way, 5-shot). args.base_class=61, args.num_classes=101, args.way=5,
args.shot=5, args.sessions=9.

Usage:
    python -m dataloader.food101.make_index_lists --root dataset \\
        --out-dir dataset/food101/index_list
"""

import argparse
import os
import random

import torchvision

BASE_CLASSES = list(range(0, 61))
NOVEL_SESSIONS = [list(range(61 + 5 * s, 66 + 5 * s)) for s in range(8)]
SHOTS_PER_CLASS = 5
SEED = 0


def _extract_paths_labels(tv_dataset):
    if hasattr(tv_dataset, "_image_files") and hasattr(tv_dataset, "_labels"):
        return [str(p) for p in tv_dataset._image_files], [int(t) for t in tv_dataset._labels]
    if hasattr(tv_dataset, "_samples"):
        paths, labels = zip(*tv_dataset._samples)
        return [str(p) for p in paths], [int(t) for t in labels]
    raise AttributeError(f"Don't know how to extract (paths, labels) from {type(tv_dataset)}")


def _paths_by_class(paths, labels):
    by_class = {}
    for p, l in zip(paths, labels):
        by_class.setdefault(l, []).append(p)
    return by_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--out-dir", type=str, default="dataset/food101/index_list")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tv = torchvision.datasets.Food101(args.root, split="train", download=True)
    paths, labels = _extract_paths_labels(tv)
    by_class = _paths_by_class(paths, labels)
    rng = random.Random(args.seed)

    base_paths = []
    for c in BASE_CLASSES:
        base_paths.extend(by_class[c])
    with open(os.path.join(args.out_dir, "session_1.txt"), "w") as f:
        for p in base_paths:
            f.write(f"{p}\n")
    print(f"session_1.txt: {len(base_paths)} samples ({len(BASE_CLASSES)} base classes, full data)")

    for s_num, session_classes in enumerate(NOVEL_SESSIONS, start=1):
        session_paths = []
        for c in session_classes:
            pool = by_class[c].copy()
            rng.shuffle(pool)
            if len(pool) < SHOTS_PER_CLASS:
                raise RuntimeError(f"class id {c} has only {len(pool)} images, fewer than SHOTS_PER_CLASS={SHOTS_PER_CLASS}.")
            session_paths.extend(pool[:SHOTS_PER_CLASS])
        fname = os.path.join(args.out_dir, f"session_{s_num + 1}.txt")
        with open(fname, "w") as f:
            for p in session_paths:
                f.write(f"{p}\n")
        print(f"session_{s_num + 1}.txt: {len(session_paths)} samples "
              f"({len(session_classes)}-way, {SHOTS_PER_CLASS}-shot: classes {session_classes})")


if __name__ == "__main__":
    main()