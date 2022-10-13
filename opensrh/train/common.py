"""Common modules for OpenSRH training and evaluation.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import math
import logging
import argparse
from shutil import copy2
from datetime import datetime
from functools import partial
from typing import Tuple, Dict, Optional, Any

import uuid

import torch
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchvision.transforms import Compose

import pytorch_lightning as pl
import torchmetrics

from opensrh.models import Classifier, MLP, resnet_backbone, vit_backbone
from opensrh.datasets.srh_dataset import (SRHClassificationDataset,
                                          SRHContrastiveDataset)
from opensrh.datasets.improc import get_srh_aug_list


def get_optimizer_func(cf: Dict[str, Any]) -> callable:
    """Return a optimizer callable based on config value"""
    lr = cf["training"]["learn_rate"]
    if cf["training"]["optimizer"] == "adamw":
        return partial(optim.AdamW, lr=lr)
    elif cf["training"]["optimizer"] == "adam":
        return partial(optim.Adam, lr=lr)
    elif cf["training"]["optimizer"] == "sgd":
        return partial(optim.SGD, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError()


def get_scheduler_func(cf: Dict[str, Any],
                       num_it_per_ep: int = 0) -> Optional[callable]:
    """Return a scheduler callable based on config value."""
    if "scheduler" not in cf["training"]:
        return None

    if cf["training"]["scheduler"]["which"] == "step_lr":
        step_size = convert_epoch_to_iter(
            cf["training"]["scheduler"]["params"]["step_unit"],
            cf["training"]["scheduler"]["params"]["step_size"], num_it_per_ep)
        return partial(optim.lr_scheduler.StepLR,
                       step_size=step_size,
                       gamma=cf["training"]["scheduler"]["params"]["gamma"])
    elif cf["training"]["scheduler"]["which"] == "cos_warmup":
        num_epochs = cf['training']['num_epochs']
        num_warmup_steps = cf['training']['scheduler']['params'][
            'num_warmup_steps']
        if isinstance(num_warmup_steps, float):
            cf['training']['scheduler']['params']['num_warmup_steps'] = int(
                num_warmup_steps * num_epochs * num_it_per_ep)
        return partial(get_cosine_schedule_with_warmup,
                       num_training_steps=num_it_per_ep * num_epochs,
                       **cf["training"]["scheduler"]["params"])
    else:
        raise NotImplementedError()


def convert_epoch_to_iter(unit: str, steps: int, num_it_per_ep: int) -> int:
    """Converts number of epochs / iterations to number of iterations."""
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    """Create cosine learn rate scheduler with linear warm up built in."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def config_loggers(exp_root):
    """Config logger for the experiments

    Sets string format and where to save.
    """

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(exp_root, 'train.log')),
                            logging.StreamHandler()
                        ],
                        force=True)
    logging.info("Exp root {}".format(exp_root))

    formatter = logging.Formatter(logging_format_str, datefmt="%H:%M:%S")
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(exp_root, 'train.log')))
    for h in logger.handlers:
        h.setFormatter(formatter)


def setup_ddp_exp_name(exp_name: str):
    if pl.utilities.rank_zero.rank_zero_only.rank != 0:
        return os.path.join(exp_name, "high_rank")
    else:
        return exp_name


def setup_output_dirs(cf: Dict, get_exp_name: callable,
                      cmt_append: str) -> Tuple[str, str, callable]:
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_name = setup_ddp_exp_name(cf["infra"]["exp_name"])
    exp_root = os.path.join(log_root, exp_name, instance_name)

    model_dir = os.path.join(exp_root, 'models')
    config_dir = os.path.join(exp_root, 'config')

    for dir_name in [model_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return exp_root, model_dir, partial(copy2, dst=config_dir)


def parse_args():
    """Get config file handle from command line argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def get_exp_name(cf):
    """Generate experiment name with a hash, time, and comments in config."""
    time = datetime.now().strftime("%b%d-%H-%M-%S")
    return "-".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])


def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4


def get_dataloaders(cf):
    """Create dataloader for cross entropy experiments."""
    train_dset = SRHClassificationDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=Compose(
            get_srh_aug_list(cf["data"]["train_augmentation"],
                             cf["data"]["rand_aug_prob"])),
        balance_patch_per_class=cf["data"]["balance_patch_per_class"])
    val_dset = SRHClassificationDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=Compose(
            get_srh_aug_list(cf["data"]["valid_augmentation"],
                             cf["data"]["rand_aug_prob"])),
        balance_patch_per_class=False)

    dataloader_callable = partial(torch.utils.data.DataLoader,
                                  batch_size=cf['training']['batch_size'],
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=get_num_worker(),
                                  persistent_workers=True)

    return dataloader_callable(train_dset, shuffle=True), dataloader_callable(
        val_dset, shuffle=False)


def get_contrastive_dataloaders(cf):
    """Create dataloader for contrastive experiments."""
    train_dset = SRHContrastiveDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=Compose(get_srh_aug_list(cf["data"]["train_augmentation"])),
        balance_patch_per_class=cf["data"]["balance_patch_per_class"])
    val_dset = SRHContrastiveDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=Compose(get_srh_aug_list(cf["data"]["valid_augmentation"])),
        balance_patch_per_class=False)

    dataloader_callable = partial(torch.utils.data.DataLoader,
                                  batch_size=cf['training']['batch_size'],
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=get_num_worker(),
                                  persistent_workers=True)

    return dataloader_callable(train_dset,
                               shuffle=True), dataloader_callable(val_dset,
                                                                  shuffle=True)


class CEBaseSystem(pl.LightningModule):
    """Abstract base lightning system for histological classification."""

    def __init__(self, cf: Dict[str, Any], nc: int, num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            bb = partial(resnet_backbone, arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "vit":
            bb = partial(vit_backbone, cf["model"]["backbone_params"])
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=nc)

        self.model = Classifier(bb, mlp)
        self.num_it_per_ep_ = num_it_per_ep

        self.criterion = CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.val_ap = torchmetrics.AveragePrecision(num_classes=nc)

    def forward(self, batch):
        return self.model(batch["image"])

    def validation_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        pred = self.model(batch["image"])
        loss = self.criterion(pred["logits"], batch["label"])

        self.val_acc.update(pred["logits"], batch["label"])
        self.val_ap.update(pred["logits"], batch["label"])
        self.val_loss.update(loss, weight=bs)

    def predict_step(self, batch, batch_idx):
        out = self.model(batch["image"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "logits": out["logits"],
            "embeddings": out["embeddings"]
        }

    def on_train_epoch_end(self):
        # compute metrics
        train_acc = self.train_acc.compute()
        train_loss = self.train_loss.compute()

        # log metrics
        self.log("train/ce", train_loss, on_epoch=True)
        self.log("train/acc", train_acc, on_epoch=True)

        # reset metrics
        self.train_loss.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        # compute metrics
        val_acc = self.val_acc.compute()
        val_loss = self.val_loss.compute()
        val_ap = self.val_ap.compute()

        # log metrics
        self.log("val/loss", val_loss, on_epoch=True)
        self.log("val/acc", val_acc, on_epoch=True)
        self.log("val/ap", val_ap, on_epoch=True)

        # reset metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_ap.reset()

    def configure_ddp(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return super().configure_ddp(*args, **kwargs)
