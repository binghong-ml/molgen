from model.lr import PolynomialDecayLR
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import moses

from model.generator2 import BaseGenerator
from data.dataset2 import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset
from data.target_data2 import Data
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize
from time import time


class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train", randomize=hparams.randomize)
        self.val_dataset = dataset_cls("valid", randomize=hparams.randomize)
        self.test_dataset = dataset_cls("test", randomize=hparams.randomize)
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
            disable_treeloc=hparams.disable_treeloc,
            disable_valencemask=hparams.disable_valencemask,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            )
        
        scheduler = PolynomialDecayLR(
            optimizer=optimizer,
            warmup_updates= self.hparams.warmup_updates, 
            tot_updates= self.hparams.tot_updates, 
            lr=self.hparams.lr,
            end_lr=self.hparams.end_lr, 
            power=1
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data[0], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data[0], ignore_index=0)[0]

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        self.sanity_checked = True
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch != 0:
            return

        num_samples = self.hparams.num_samples if self.sanity_checked else 1024
        max_len = self.hparams.max_len if self.sanity_checked else 10
        maybe_smiles_list, tokens_list, errors = self.sample(num_samples, max_len)

        smiles_list = []
        for maybe_smiles, tokens, error in zip(maybe_smiles_list, tokens_list, errors):
            smiles, _ = canonicalize(maybe_smiles)
            if smiles is None:
                self.logger.experiment["invalid/maybe_smiles"].log(f"{self.current_epoch}, {maybe_smiles}")
                self.logger.experiment["invalid/tokens"].log(f"{self.current_epoch}, {tokens}")
                self.logger.experiment["invalid/error"].log(f"{self.current_epoch}, {error}")
            else:
                smiles_list.append(smiles)
                self.logger.experiment["valid/maybe_smiles"].log(f"{self.current_epoch}, {maybe_smiles}")
                self.logger.experiment["valid/tokens"].log(f"{self.current_epoch}, {tokens}")
                self.logger.experiment["valid/error"].log(f"{self.current_epoch}, {error}")
                
        unique_smiles_set = set(smiles_list)
        novel_smiles_set = unique_smiles_set - self.train_smiles_set

        statistics = dict()
        statistics["sample/valid"] = float(len(smiles_list)) / num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / num_samples
        statistics["sample/novel"] = float(len(novel_smiles_set)) / num_samples

        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    def sample(self, num_samples, max_len, verbose=False):
        offset = 0
        maybe_smiles_list = []
        tokens_list = []
        errors = []
        tic = time()

        #self.to(0)
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                data_list = self.model.decode(cur_num_samples, max_len=max_len, device=self.device)

            tokens_list += ["".join(data.tokens) for data in data_list]
            errors += [data.error for data in data_list]
            
            for data in data_list:
                if data.error is None:
                    smiles = data.to_smiles()
                    maybe_smiles_list.append(smiles)
                else:
                    maybe_smiles_list.append(None)

                errors.append(data.error)
            
            if verbose:
                elapsed = time() - tic
                print(f"{len(maybe_smiles_list)} / {num_samples}, elaspsed: {elapsed}")

        return maybe_smiles_list, tokens_list, errors

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="zinc")

        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        #parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument("--tot_updates", type=int, default=500000)
        parser.add_argument("--warmup_updates", type=int, default=10000)


        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--num_samples", type=int, default=128)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        parser.add_argument("--test_num_samples", type=int, default=10000)

        parser.add_argument("--randomize", action="store_true")
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    neptune_logger = NeptuneLogger(project="sungsahn0215/molgen", close_after_fit=False, source_files="**/*.py")
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run["sys/tags"].add(hparams.tag.split("_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("../resource/checkpoint/", hparams.tag), monitor="validation/loss/total", mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=hparams.gradient_clip_val,
        resume_from_checkpoint=hparams.resume_from_checkpoint_path,
    )
    trainer.fit(model)

    model = model.to(0)
    model.eval()
    with torch.no_grad():
        smiles_list, _, _ = model.sample(hparams.test_num_samples, hparams.max_len, verbose=True)
    
    smiles_list = [smiles for smiles in smiles_list if smiles is not None]
    smiles_list_path = os.path.join("../resource/checkpoint/", hparams.tag, "test.txt")
    Path(smiles_list_path).write_text("\n".join(smiles_list))

    metrics = moses.get_all_metrics(
        smiles_list, n_jobs=8, 
        device="cuda:0", 
        train=model.train_dataset.smiles_list, 
        test=model.test_dataset.smiles_list,
        )
    print(metrics)
    for key in metrics:
        neptune_logger.experiment[f"moses_metric/{key}"] = metrics[key]
