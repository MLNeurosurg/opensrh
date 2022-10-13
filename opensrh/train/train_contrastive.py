"""Contrastive learning experiment training script.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any

import torch

import pytorch_lightning as pl
import torchmetrics

from opensrh.models import MLP, resnet_backbone, ContrastiveLearningNetwork, vit_backbone
from opensrh.train.common import (setup_output_dirs, parse_args, get_exp_name,
                                  get_contrastive_dataloaders, config_loggers,
                                  get_optimizer_func, get_scheduler_func)
from opensrh.losses.supcon import SupConLoss


class ContrastiveSystem(pl.LightningModule):
    """Lightning system for contrastive learning experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
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
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)
        self.criterion = SupConLoss()
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.num_it_per_ep_ = num_it_per_ep

    def predict_step(self, batch, batch_idx):
        out = self.model.bb(batch["image"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=False,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=False,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # check if use a learn rate scheduler
        sched_func = get_scheduler_func(self.cf_, self.num_it_per_ep_)
        if not sched_func:
            return opt

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sched_func(opt),
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config

    def configure_ddp(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return super().configure_ddp(*args, **kwargs)


class SimCLRSystem(ContrastiveSystem):
    """Lightning system for SimCLR experiment"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)

    def forward(self, data):
        return torch.cat([self.model(x) for x in data["image"]], dim=1)

    def training_step(self, batch, batch_idx):
        pred = torch.cat([self.model(x) for x in batch["image"]], dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        loss = self.criterion(pred_gather)
        bs = batch["image"][0].shape[0]
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs)
        self.train_loss.update(loss, weight=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        bs = batch["image"][0].shape[0]
        pred = torch.cat([self.model(x) for x in batch["image"]], dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        loss = self.criterion(pred_gather)
        self.val_loss.update(loss, weight=bs)


class SupConSystem(ContrastiveSystem):
    """Lightning system for SupCon experiment"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)

    def forward(self, data):
        return torch.cat([self.model(x) for x in data["image"]], dim=1)

    def training_step(self, batch, batch_idx):
        pred = torch.cat([self.model(x) for x in batch["image"]], dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)
        bs = batch["image"][0].shape[0]
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs)
        self.train_loss.update(loss, weight=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        bs = batch["image"][0].shape[0]
        pred = torch.cat([self.model(x) for x in batch["image"]], dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)
        self.val_loss.update(loss, weight=bs)


def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get dataloaders
    train_loader, valid_loader = get_contrastive_dataloaders(cf)
    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()

    if cf["training"]["objective"] == "supcon":
        system_func = SupConSystem
    elif cf["training"]["objective"] == "simclr":
        system_func = SimCLRSystem
    else:
        raise NotImplementedError()

    ce_exp = system_func(cf, num_it_per_ep)

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        filename="ckpt-epoch{epoch}-loss{val/contrastive_manualepoch:.2f}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=-1,
                         default_root_dir=exp_root,
                         strategy=pl.strategies.DDPStrategy(
                             find_unused_parameters=False, static_graph=True),
                         logger=logger,
                         log_every_n_steps=10,
                         callbacks=[epoch_ckpt, lr_monitor],
                         max_epochs=cf["training"]["num_epochs"])
    trainer.fit(ce_exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
