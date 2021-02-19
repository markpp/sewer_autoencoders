import os, sys
import numpy as np

import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    """
    Test script

    Command:
        python test.py
    """

    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    # init default datamodule
    dm = CIFAR10DataModule(batch_size=128)
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)

    from pl_bolts.models.self_supervised.byol.models import SiameseArm

    online_network = SiameseArm()
    online_network = torch.load('trained_models/online_network.pt')
    online_network.eval()

    dm.setup()

    with torch.no_grad():

        sample_idx = 0
        for batch_id, batch in enumerate(dm.val_dataloader()):
            x, y = batch
            x = x[0]
            print(y.size())
            z, _, _ = online_network(x)
            print(z.size())
            break
