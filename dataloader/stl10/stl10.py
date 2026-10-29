import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision


class STL10(Dataset):
    """
    STL10 wrapped to match this repo's Dataset-class interface, including
    multi-crop MoCo-style output (matches dataloader/miniimagenet's
    __getitem__ pattern: classify_image + multi_crop when multi_train=True).

      train=True,  base_sess=True,  index=<array of class ids>  -> base
          session: ALL training samples of those classes, multi-crop
          output for MoCo contrastive training.
      train=True,  base_sess=False, index_path=<txt file path>  -> an
          incremental session: EXACTLY the sample indices listed in the
          file (one integer per line, indexing into the STL10 train split).
      train=False, index=<array of class ids>                   -> test
          set for evaluation: ALL test samples of those classes.

    Benchmark definition:
      STL10 class order is torchvision's default:
        0 airplane, 1 bird, 2 car, 3 cat, 4 deer,
        5 dog, 6 horse, 7 monkey, 8 ship, 9 truck
      Base classes (session 0):      0-5  (airplane..dog),   full data
      Session 1 novel (2-way,5-shot): 6,7  (horse, monkey)
      Session 2 novel (2-way,5-shot): 8,9  (ship, truck)
      args.base_class=6, args.way=2, args.shot=5, args.num_classes=10,
      args.sessions=3 (session 0=base, sessions 1-2=incremental).

    STL10 has no per-image files (torchvision stores it as one binary blob
    indexed by position, native shape (3,96,96) channel-first), so
    index_path files here list raw integer indices into the STL10 train
    split instead of image paths (see dataloader/stl10/make_index_lists.py).
    """

    def __init__(self, root='./data', train=True,
                 index_path=None, index=None, base_sess=None, download=True,
                 crop_transform=None, secondary_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)

        # multi_train mirrors MiniImageNet's convention: True only for the
        # base-session training set, where MoCo-style multi-crop views are
        # needed. Incremental sessions and the test set use a single view.
        self.multi_train = bool(train and base_sess and crop_transform is not None)

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        split = 'train' if train else 'test'
        stl = torchvision.datasets.STL10(root=self.root, split=split, download=download)
        # torchvision STL10: .data is uint8 (N, 3, 96, 96) channel-first, .labels is int (N,)
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
        img = self.data[i]                     # (3, 96, 96) uint8, channel-first (BCHW-style, per-sample CHW)
        img = np.transpose(img, (1, 2, 0))      # -> (96, 96, 3) HWC, required by PIL.Image.fromarray
        image = Image.fromarray(img)
        target = int(self.targets[i])

        if self.multi_train:
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            total_image = classify_image + multi_crop
        else:
            total_image = self.transform(image)

        return total_image, target