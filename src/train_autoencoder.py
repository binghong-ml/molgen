import os
import argparse
from joblib import Parallel, delayed
import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from model.translator import BaseTranslator
from data.dataset import ZincAutoEncoderDataset
from data.data import SourceData, TargetData
from util import compute_sequence_cross_entropy, compute_sequence_accuracy, canonicalize
from score import _raw_plogp_improvement

class BaseTranslatorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseTranslatorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        self.train_dataset = ZincAutoEncoderDataset("train")
        self.val_dataset = ZincAutoEncoderDataset("valid")
        self.tokenizer = self.train_dataset.tokenizer

        def collate(data_list):
            src, tgt = zip(*data_list)
            src = SourceData.collate(src)
            tgt = TargetData.collate(tgt)
            return src, tgt

        self.collate = collate

    def setup_model(self, hparams):
        self.model = BaseTranslator(
            self.tokenizer,
            hparams.num_layers,
            hparams.emb_size,
            hparams.nhead,
            hparams.dim_feedforward,
            hparams.dropout,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data, batch_idx):
        loss, statistics = 0.0, dict()
        src, tgt = batched_data

        # decoding
        logits = self.model(src, tgt)
        recon_loss = compute_sequence_cross_entropy(logits, tgt[0])
        loss += recon_loss

        element_acc, sequence_acc = compute_sequence_accuracy(logits, tgt[0])

        statistics["loss/total"] = loss
        statistics["acc/element"] = element_acc
        statistics["acc/sequence"] = sequence_acc
        
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        self.sanity_checked = True
        
        loss, statistics = self.shared_step(batched_data, batch_idx)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss
        
    def validation_step(self, batched_data, batch_idx):
        
        loss, statistics = self.shared_step(batched_data, batch_idx)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="moses_yessinglebond")

        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--num_repeats", type=int, default=20)

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTranslatorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseTranslatorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_from_checkpoint(hparams.load_checkpoint_path)

    neptune_logger = NeptuneLogger(project="sungsahn0215/molgen", close_after_fit=False)
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run["sys/tags"].add(hparams.tag.split("_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("../resource/checkpoint/", hparams.tag), monitor="validation/loss/total", mode="min",
    )
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)
