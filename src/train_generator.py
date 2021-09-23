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
from moses.utils import disable_rdkit_log, enable_rdkit_log

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset
from data.target_data import Data
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        
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
        
        return [optimizer]

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
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        smiles_list, results = self.sample(num_samples)

        #
        if not self.trainer.sanity_checking:
            for (smiles, result) in zip(smiles_list, results):
                self.logger.experiment[f"sample/smiles/{self.current_epoch:03d}"].log(smiles)
                self.logger.experiment[f"sample/result/{self.current_epoch:03d}"].log(result)

        #
        valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_set = unique_smiles_set - self.train_smiles_set
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_set)) / len(valid_smiles_list)

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)
        
        if self.hparams.eval_moses and len(valid_smiles_list) > 0:
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers, 
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            for key in moses_statistics:
                self.log(f"sample/moses/{key}", moses_statistics[key])

    def sample(self, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                data_list = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results

    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        #
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)
        
        #
        parser.add_argument("--randomize", action="store_true")
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        
        #
        parser.add_argument("--lr", type=float, default=1e-4)

        #
        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=10)
        parser.add_argument("--num_samples", type=int, default=10000)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        parser.add_argument("--eval_moses", action="store_true")

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsIm"
        "FwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ==",
        project="sungsahn0215/molgen",
        source_files="**/*.py"
        )
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
        resume_from_checkpoint=hparams.resume_from_checkpoint_path,
    )
    trainer.fit(model)