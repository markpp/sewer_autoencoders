from argparse import ArgumentParser
import os, sys
import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer, loggers
import torch.nn.functional as F

from model import BYOL

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    seed_everything(1234)



    sys.path.append('../datasets/sewer/')
    from datamodule import SewerDataModule
    dm = SewerDataModule(data_dir='/home/datasets/sewer/',
                         batch_size=64,
                         image_size=256)

    model = BYOL()

    # finetune in real-time
    #online_eval = SSLOnlineEvaluator(dataset=args.dataset, z_dim=2048)

    trainer = pl.Trainer(gpus=1, max_steps=300000)

    trainer.fit(model, datamodule=dm)


    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(model.online_network, os.path.join(output_dir,"online_network.pt"))
