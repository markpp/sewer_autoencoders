import os, sys
import numpy as np
import cv2

import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    """
    Test script

    Command:
        python test.py
    """

    """
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    # init default datamodule
    dm = CIFAR10DataModule(batch_size=4096)
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)
    """


    sys.path.append('../datasets/sewer/')
    from datamodule import SewerDataModule
    dm = SewerDataModule(data_dir='/home/datasets/sewer/',
                         batch_size=16,
                         image_size=256)

    dm.setup()

    """
    from pl_bolts.models.self_supervised.byol.models import SiameseArm

    online_network = SiameseArm()
    online_network = torch.load('trained_models/online_network.pt')
    online_network.eval()
    """

    with torch.no_grad():

        sample_idx = 0
        for batch_id, batch in enumerate(dm.val_dataloader()):
            imgs, imgs_ = batch
            for img, img_ in zip(imgs,imgs_):
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                img_ = img_.mul(255).permute(1, 2, 0).byte().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                cv2.imwrite("output/{}_img.jpg".format(sample_idx),img)
                cv2.imwrite("output/{}_img_.jpg".format(sample_idx),img_)
                sample_idx = sample_idx + 1
            break


            '''
            z, _, _ = online_network(x)
            np.save("y.npy",y.numpy())
            np.save("z.npy",z.numpy())
            break
            '''
