import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os, sys
from glob import glob
import cv2

#sys.path.append('../')
from dataset import SewerDataset


import albumentations as Augment

def basic_transforms(img_height, img_width, image_pad=0):
    return Augment.Compose([#Augment.ToGray(p=1.0),
                            Augment.LongestMaxSize(max_size=360, always_apply=True),
                            Augment.Rotate(limit=360, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
                            #Augment.Resize(img_height+image_pad, img_width+image_pad, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            #Augment.RandomResizedCrop(img_height, img_width, scale=(1.0, 1.0), always_apply=True),
                            Augment.HorizontalFlip(p=0.5),
                            Augment.RandomBrightnessContrast(p=1.0),
                            ], p=1)#ToTensor()

class SewerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = SewerDataset(os.path.join(self.data_dir,'imgs/train'), transform=basic_transforms(img_height=self.image_size,img_width=self.image_size))
            self.data_val = SewerDataset(os.path.join(self.data_dir,'imgs/val'), transform=basic_transforms(img_height=self.image_size,img_width=self.image_size))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=12)


if __name__ == '__main__':

    dm = SewerDataModule(data_dir='/home/markpp/datasets/sewer/',
                         batch_size=16,
                         image_size=256)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if not os.path.exists(output_root):
        #shutil.rmtree(output_root)
        os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        imgs = batch
        for img in imgs:
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
        if batch_id > 0:
            break
