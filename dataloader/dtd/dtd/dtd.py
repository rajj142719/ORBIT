import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision


def _extract_paths_labels(tv_dataset):
    """See food101.py's _extract_paths_labels for rationale."""
    if hasattr(tv_dataset, "_image_files") and hasattr(tv_dataset, "_labels"):
        return [str(p) for p in tv_dataset._image_files], [int(t) for t in tv_dataset._labels]
    if hasattr(tv_dataset, "_samples"):
        paths, labels = zip(*tv_dataset._samples)
        return [str(p) for p in paths], [int(t) for t in labels]
    if hasattr(tv_dataset, "samples"):
        paths, labels = zip(*tv_dataset.samples)
        return [str(p) for p in paths], [int(t) for t in labels]
    raise AttributeError(
        f"Don't know how to extract (paths, labels) from {type(tv_dataset)} -- "
        f"its internal attribute names may have changed in this torchvision "
        f"version. Inspect vars(tv_dataset).keys() and update _extract_paths_labels()."
    )


class DTD(Dataset):
    """
    Describable Textures Dataset (47 classes) wrapped to match this repo's
    Dataset-class interface -- same calling convention as
    MiniImageNet/FGVCAircraft/Food101/GTSRB.

    torchvision's DTD is split into 10 fixed partitions, each with its own
    train/val/test split (~40 images/class per split within a partition).
    To get a large-enough trainval pool, this class merges partition 1's
    'train' and 'val' splits into one "trainval" pool for training (same
    idea as FGVCAircraft's trainval merge), and uses partition 1's 'test'
    split for evaluation. Only partition 1 is used -- not all 10 -- for
    simplicity and to match this repo's convention of a single deterministic
    train/test split per dataset.

    Split (47 classes total):
      Base classes (session 0):          class ids 0-26,  full trainval data
      Novel sessions 1-4 (5-way,5-shot): class ids 27-31, 32-36, 37-41, 42-46
      args.base_class=27, args.num_classes=47, args.way=5, args.shot=5,
      args.sessions=5.

    train=True,  base_sess=True,  index=<array of class ids>  -> base session
    train=True,  base_sess=False, index_path=<txt file path>  -> incremental
        session: exact image paths listed in the file (one per line).
    train=False, index=<array of class ids>                   -> test set
    """

    PARTITION = 1

    def __init__(self, root='./dataset', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None,
                 download=True):
        self.root = os.path.expanduser(root)
        self.transform = None
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)
        self.multi_train = bool(train and base_sess and crop_transform is not None)

        if train:
            # merge 'train' + 'val' splits of partition 1 into one trainval pool
            tv_train = torchvision.datasets.DTD(root=self.root, split='train',
                                                 partition=self.PARTITION, download=download)
            tv_val = torchvision.datasets.DTD(root=self.root, split='val',
                                               partition=self.PARTITION, download=download)
            data_train, targets_train = _extract_paths_labels(tv_train)
            data_val, targets_val = _extract_paths_labels(tv_val)
            self.data = data_train + data_val
            self.targets = targets_train + targets_val
        else:
            tv_test = torchvision.datasets.DTD(root=self.root, split='test',
                                                partition=self.PARTITION, download=download)
            self.data, self.targets = _extract_paths_labels(tv_test)

        self.data2label = {p: t for p, t in zip(self.data, self.targets)}

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, data2label, index_path):
        lines = [x.strip() for x in open(index_path, 'r').readlines() if x.strip()]
        data_tmp, targets_tmp = [], []
        for img_path in lines:
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        targets_arr = np.array(targets)
        data_tmp, targets_tmp = [], []
        for i in index:
            ind_cl = np.where(targets_arr == i)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, target = self.data[i], self.targets[i]
        image = Image.open(path).convert('RGB')

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