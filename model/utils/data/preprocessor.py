from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import random
import math
from PIL import Image
import torchvision.transforms.functional as TF
from model.utils.data import transforms as T


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img2 = img.copy()

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img2)
        else:
            raise NotImplementedError

        return img1, img2, fname, pid, camid, index
