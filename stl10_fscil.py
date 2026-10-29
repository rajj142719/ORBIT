import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision


class STL10(Dataset):
    """
    STL10 wrapped to match this repo's Dataset-class interface (same
    calling convention as CIFAR100 in dataloader/cifar100/cifar.py):

      train=True,  base_sess=True,  index=<array of class ids>  -> base
          session: ALL training samples of those classes.
      train=True,  base_sess=False, index_path=<txt file path>  -> an
          incremental session: EXACTLY the sample indices listed in the
          file (one integer per line, indexing into the STL10 train split).
      train=False, index=<array of class ids>                   -> test
          set for evaluation: ALL test samples of those classes. Pass the
          cumulative class list (base + every novel class seen so far) to
          get the standard FSCIL cumulative-evaluation protocol.

    Benchmark definition (rationale in stl10_fscil.py -- kept identical
    here so results from that script and this training pipeline agree):
      STL10 class order is torchvision's default:
        0 airplane, 1 bird, 2 car, 3 cat, 4 deer,
        5 dog, 6 horse, 7 monkey, 8 ship, 9 truck
      Base classes (session 0):      0-5  (airplane..dog),   full data
      Session 1 novel (2-way,5-shot): 6,7  (horse, monkey)
      Session 2 novel (2-way,5-shot): 8,9  (ship, truck)
      args.base_class=6, args.way=2, args.shot=5, args.num_classes=10,
      args.sessions=3 (session 0=base, sessions 1-2=incremental).

    Difference from CUB200/MiniImageNet in this repo: those use
    ImageFolder-style file paths, so their index_path files list image
    paths. STL10 has no per-image files -- torchvision stores it as one
    binary blob indexed by position -- so index_path files here list raw
    integer indices into the STL10 train split instead (see
    dataloader/stl10/make_index_lists.py, which generates them).
    """

    def __init__(self, root='./data', train=True, transform=None,
                 index_path=None, index=None, base_sess=None, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        split = 'train' if train else 'test'
        stl = torchvision.datasets.STL10(root=self.root, split=split, download=download)
        # torchvision STL10: .data is uint8 (N, 3, 96, 96), .labels is int (N,)
        self.data = stl.data
        self.targets = np.asarray(stl.labels, dtype=np.int64)

        if train:
            if base_sess:
                if index is None:
                    raise ValueError("STL10 base session requires `index` (array of class ids).")
                self.SelectfromClasses(index)
            else:
                if index_path is None:
                    raise ValueError("STL10 incremental session requires `index_path`.")
                self.SelectfromTxt(index_path)
        else:
            if index is None:
                raise ValueError("STL10 test set requires `index` (cumulative array of class ids).")
            self.SelectfromClasses(index)

    def SelectfromClasses(self, index):
        """Keep every sample whose label is in `index` (array-like of class ids)."""
        index = np.asarray(index)
        mask = np.isin(self.targets, index)
        self.data = self.data[mask]
        self.targets = self.targets[mask]

    def SelectfromTxt(self, index_path):
        """Keep exactly the sample positions listed in `index_path`
        (one integer per line, indexing into the original STL10 train split)."""
        with open(index_path) as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        indices = np.asarray(indices)
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = self.data[i]                     # (3, 96, 96) uint8, channel-first
        img = np.transpose(img, (1, 2, 0))      # -> (96, 96, 3) for PIL
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.targets[i])
        return img, target