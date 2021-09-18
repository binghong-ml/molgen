from collections import defaultdict
import os
import argparse
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from model.translator import BaseTranslator
from data.dataset import LogP04Dataset, LogP06Dataset, DRD2Dataset, QEDDataset
from data.source_data import Data as SourceData
from data.target_data import Data as TargetData
from util import compute_sequence_cross_entropy, compute_sequence_accuracy, canonicalize


class BaseTranslatorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseTranslatorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        dataset_cls = {"logp04": LogP04Dataset, "logp06": LogP06Dataset, "drd2": DRD2Dataset, "qed": QEDDataset}.get(
            self.hparams.dataset_name
        )
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")

        def train_collate(data_list):
            src, tgt = zip(*data_list)
            src = SourceData.collate(src)
            tgt = TargetData.collate(tgt)
            return src, tgt

        self.train_collate = train_collate

        def eval_collate(data_list):
            src, src_smiles_list = zip(*data_list)
            src = SourceData.collate(src)
            return src, src_smiles_list

        self.eval_collate = eval_collate

    def setup_model(self, hparams):
        self.model = BaseTranslator(
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
            collate_fn=self.train_collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collate,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collate,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def training_step(self, batched_data, batch_idx):
        self.sanity_checked = True

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
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        statistics = dict()
        src, src_smiles_list = batched_data
        max_len = self.hparams.max_len if self.sanity_checked else 10

        with torch.no_grad():
            tgt_data_list = self.model.decode(src, max_len=max_len, device=self.device)

        smiles_list = []
        for tgt_data, src_smiles in zip(tgt_data_list, src_smiles_list):
            if tgt_data.error is None:
                try:
                    maybe_smiles = tgt_data.to_smiles()
                    error = None
                except Exception as e:
                    maybe_smiles = ""
                    error = e

            else:
                maybe_smiles = ""
                error = tgt_data.error

            if canonicalize(maybe_smiles) is None:
                self.logger.experiment["invalid_smiles"].log(
                    f"{self.current_epoch}, {src_smiles}, {maybe_smiles}, {error}"
                    )
            else:
                smiles_list.append(maybe_smiles)

        statistics["validation/valid"] = float(len(smiles_list)) / src[0].size(0)
        statistics["validation/unique"] = (
            float(len(set(smiles_list))) / len(smiles_list) if len(smiles_list) > 0 else 0.0
        )
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batched_data, batch_idx):
        src, src_smiles_list = batched_data
        src_smiles2tgt_smiles_list = defaultdict(list)
        invalid = 0
        for _ in range(self.hparams.num_repeats):
            with torch.no_grad():
                tgt_data_list = self.model.decode(src, max_len=self.hparams.max_len, device=self.device)

            for src_smiles, tgt_data in zip(src_smiles_list, tgt_data_list):
                maybe_smiles = tgt_data.to_smiles()
                if canonicalize(maybe_smiles) is None:
                    invalid += 1

                src_smiles2tgt_smiles_list[src_smiles].append(maybe_smiles)

        print(float(invalid) / self.hparams.num_repeats / src[0].size(0))

        dict_path = os.path.join(self.hparams.checkpoint_dir, "test_pairs.txt")
        for src_smiles in src_smiles_list:
            with Path(dict_path).open("a") as fp:
                fp.write(" ".join([src_smiles] + src_smiles2tgt_smiles_list[src_smiles]) + "\n")

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="logp04")

        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--eval_batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--num_repeats", type=int, default=20)
        parser.add_argument("--max_len", type=int, default=250)

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTranslatorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseTranslatorLightningModule(hparams)
    # if hparams.load_checkpoint_path != "":
    #model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    if not hparams.debug:
        logger = NeptuneLogger(project="sungsahn0215/molgen", close_after_fit=False)
        logger.run["params"] = vars(hparams)
        logger.run["sys/tags"].add(hparams.tag.split("_"))
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join("../resource/checkpoint/", hparams.tag), monitor="validation/valid", mode="max",
        )
        callbacks = [checkpoint_callback]
    else:
        logger = None
        callbacks = []

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)

    if hparams.max_epochs > 0:
        model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    model.eval()
    trainer.test(model)
