import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os
from glob import glob
import cv2
import random

class SewerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.png'))])
        if not len(self.image_list)>0:
            print("did not find any files")

        self.transform = transform

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #self.image_h, self.image_w, _ = img.shape
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])

        #img = Image.fromarray(img)

        if self.transform:
            sample = self.transform(**{'image':img})
            x = sample["image"]
            sample = self.transform(**{'image':img})
            x_ = sample["image"]

        x = x.transpose((2, 0, 1))
        x_ = x_.transpose((2, 0, 1))


        return torch.as_tensor(x)/255.0, torch.as_tensor(x_)/255.0

    def __len__(self):
        return len(self.image_list)
