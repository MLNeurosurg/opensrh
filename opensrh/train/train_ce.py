"""Cross entropy experiment training script.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from typing import Dict, Any

import torch

import pytorch_lightning as pl

from opensrh.train.common import (setup_output_dirs, parse_args, get_exp_name,
                                  get_dataloaders, config_loggers,
                                  get_optimizer_func, get_scheduler_func,
                                  CEBaseSystem)


class CESystem(CEBaseSystem):
    """Lightning system for histological classification experiment."""

    def __init__(self, cf: Dict[str, Any], nc: int, num_it_per_ep: int):
        super().__init__(cf, nc, num_it_per_ep)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch["image"])
        loss = self.criterion(pred["logits"], batch["label"])

        bs = batch["image"].shape[0]
        acc = self.train_acc(pred["logits"], batch["label"])
        self.train_loss.update(loss, weight=bs)

        self.log("train/ce", loss, on_step=True, batch_size=bs)
        self.log("train/acc", acc, on_step=True, batch_size=bs)

        return loss

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


def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get dataloaders
    train_loader, valid_loader = get_dataloaders(cf)
    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    nc = len(train_loader.dataset.classes_)
    num_it_per_ep = len(train_loader)
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()

    ce_exp = CESystem(cf, nc, num_it_per_ep)

    # load lightning checkpoint to initialize model weights
    if "backbone_checkpoint" in cf["training"]:
        ckpt_dict = torch.load(cf["training"]["backbone_checkpoint"],
                               map_location="cpu")
        state_dict = {
            k.removeprefix("model.bb."): ckpt_dict["state_dict"][k]
            for k in ckpt_dict["state_dict"] if "model.bb" in k
        }
        ce_exp.model.bb.load_state_dict(state_dict)

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
        filename="ckpt-epoch{epoch}-acc{val/acc:.2f}",
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
