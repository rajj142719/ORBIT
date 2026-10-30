#!/usr/bin/env python
"""
make_index_lists.py (DTD)

Generates FSCIL index-list files at:
    <dataroot>/dtd/index_list/session_2.txt .. session_5.txt

Uses partition 1's merged train+val split as the trainval pool (matches
dtd.py's DTD class -- see that file's docstring for rationale).

Split (47 classes total): base 0-26, novel sessions 27-31,...,42-46
(4 x 5-way, 5-shot). args.base_class=27, args.num_classes=47, args.way=5,
args.shot=5, args.sessions=5.

Usage:
    python -m dataloader.dtd.make_index_lists --root dataset \\
        --out-dir dataset/dtd/index_list
"""

import argparse
import os
import random

import torchvision

BASE_CLASSES = list(range(0, 27))
NOVEL_SESSIONS = [list(range(27 + 5 * s, 32 + 5 * s)) for s in range(4)]
SHOTS_PER_CLASS = 5
SEED = 0
PARTITION = 1


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
    parser.add_argument("--out-dir", type=str, default="dataset/dtd/index_list")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tv_train = torchvision.datasets.DTD(args.root, split="train", partition=PARTITION, download=True)
    tv_val = torchvision.datasets.DTD(args.root, split="val", partition=PARTITION, download=True)
    p_train, l_train = _extract_paths_labels(tv_train)
    p_val, l_val = _extract_paths_labels(tv_val)
    paths = p_train + p_val
    labels = l_train + l_val
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