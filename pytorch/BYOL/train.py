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

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        args.num_classes = dm.num_classes

    elif args.dataset == 'stl10':
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed

        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    elif args.dataset == 'imagenet2012':
        dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    model = BYOL(**args.__dict__)

    # finetune in real-time
    online_eval = SSLOnlineEvaluator(dataset=args.dataset, z_dim=2048, num_classes=dm.num_classes)

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, max_steps=300000, resume_from_checkpoint='trained_models/epoch=431-step=8639.ckpt', callbacks=[online_eval])

    trainer.fit(model, datamodule=dm)


    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(model.online_network, os.path.join(output_dir,"online_network.pt"))
