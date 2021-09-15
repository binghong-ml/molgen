import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import moses

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, PolymerDataset
from data.util import RING_TOKENS, Data, get_value_id
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize


class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        dataset_cls = {"zinc": ZincDataset, "moses": MosesDataset, "polymer": PolymerDataset}.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        val_sequences, ring_sequences, distance_squares = batched_data

        # decoding
        logits = self.model(val_sequences, ring_sequences, distance_squares)
        
        targets = val_sequences * len(RING_TOKENS) + ring_sequences
        targets[val_sequences==get_value_id("<pad>")] = 0
        loss = compute_sequence_cross_entropy(logits, targets, ignore_index=0)
        
        statistics["loss/total"] = loss
        
        pred = torch.argmax(logits[:, :-1], dim=-1)
        val_pred = pred % len(RING_TOKENS)
        ring_pred = pred // len(RING_TOKENS)        
        
        statistics["acc/elem/value"] = (
            ((val_pred == val_sequences[:, 1:])[val_sequences[:, 1:] != 0]).float().mean()
        )
        statistics["acc/elem/ring"] = (
            ((ring_pred == ring_sequences[:, 1:])[ring_sequences[:, 1:] != 0]).float().mean()
        )
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
        num_samples = self.hparams.num_samples if self.sanity_checked else 256
        max_len = self.hparams.max_len if self.sanity_checked else 100
        maybe_smiles_list, errors = self.sample(num_samples, max_len)

        for error in errors:
            self.logger.experiment["error"].log(f"{self.current_epoch}, {error}")

        smiles_list = []
        for maybe_smiles in maybe_smiles_list:
            self.logger.experiment["maybe_smiles"].log(f"{self.current_epoch}, {maybe_smiles}")            

            smiles, error = canonicalize(maybe_smiles)
            if smiles is None:
                self.logger.experiment["invalid_smiles"].log(f"{self.current_epoch}, {maybe_smiles}, {error}")
            else:
                self.logger.experiment["valid_smiles"].log(f"{self.current_epoch}, {maybe_smiles}")
                smiles_list.append(smiles)

        unique_smiles_set = set(smiles_list)
        novel_smiles_set = unique_smiles_set - self.train_smiles_set

        statistics = dict()
        statistics["sample/valid"] = float(len(smiles_list)) / num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / num_samples if len(smiles_list) > 0 else 0.0
        statistics["sample/novel"] = float(len(novel_smiles_set)) / num_samples if len(smiles_list) > 0 else 0.0

        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    def sample(self, num_samples, max_len, verbose=False):
        offset = 0
        maybe_smiles_list = []
        errors = []
        while offset < num_samples: 
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            with torch.no_grad():
                data_list = self.model.decode(cur_num_samples, max_len=max_len, device=self.device)

            for data in data_list:
                if data.error is None:
                    try:
                        smiles = data.to_smiles()
                        maybe_smiles_list.append(smiles)
                    except Exception as e:
                        errors.append(e)
                else:
                    errors.append(data.error)

            if verbose:
                print(f"{len(maybe_smiles_list)} / {num_samples}")
        
        return maybe_smiles_list, errors

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="zinc")

        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--num_samples", type=int, default=256)
        parser.add_argument("--sample_batch_size", type=int, default=256)
        parser.add_argument("--test_num_samples", type=int, default=30000)
        
        return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseGeneratorLightningModule(hparams)
    #model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    neptune_logger = NeptuneLogger(project="sungsahn0215/molgen", close_after_fit=False)
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run["sys/tags"].add(hparams.tag.split("_"))
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath=os.path.join("../resource/checkpoint/", hparams.tag), monitor="validation/loss/total", mode="min",
    #)
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        #callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)

    model.eval()
    with torch.no_grad():
        smiles_list, _ = model.sample(hparams.test_num_samples, hparams.max_len, verbose=True)

    smiles_list_path = os.path.join("../resource/checkpoint/", hparams.tag, "test.txt")
    Path(smiles_list_path).write_text("\n".join(smiles_list))

    metrics = moses.get_all_metrics(smiles_list, n_jobs=8, device="cuda:0", test=model.test_dataset.smiles_list)
    print(metrics)
    for key in metrics:
        neptune_logger.experiment[f"moses/{key}"] = metrics[key]