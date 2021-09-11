import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import moses

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, PolymerDataset, collate
from data.util import node_feature_names, edge_feature_names

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
        
        self.collate = collate

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            hparams.num_encoder_layers,
            hparams.num_decoder_layers,
            hparams.emb_size,
            hparams.nhead,
            hparams.dim_feedforward,
            hparams.dropout,
            hparams.logit_hidden_dim,
            hparams.vq_vocab_size,
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
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        batched_node_data, batched_edge_data = batched_data
        
        node_logits, edge_logits, vq_loss = self.model(batched_node_data, batched_edge_data)
            
        statistics[f"loss/vq"] = vq_loss

        correct_total = 1.0
        loss_total = 0.0
        loss_total += 0.01 * vq_loss
        for key in node_feature_names + edge_feature_names:
            if key in node_feature_names:
                batch_size = node_logits[key].size(0)
                logits = node_logits[key].reshape(-1, node_logits[key].size(-1))
                targets = batched_node_data[key].reshape(-1)
            else:
                batch_size = edge_logits[key].size(0)
                logits = edge_logits[key].reshape(-1, edge_logits[key].size(-1))
                targets = batched_edge_data[key].reshape(-1)

            loss = F.cross_entropy(logits, targets, ignore_index=0) 
            
            preds = torch.argmax(logits, -1)
            correct = preds == targets
            correct[targets == 0] = True
            elem_acc = correct[targets != 0].float().mean()
            sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()
            
            correct_total = correct_total * correct.view(batch_size, -1).all(dim=1).float()
            if key in node_feature_names:
                statistics[f"loss/node_{key}"] = loss
                statistics[f"elem_acc/node_{key}"] = elem_acc 
                statistics[f"seq_acc/node_{key}"] = sequence_acc
            else:
                statistics[f"loss/edge_{key}"] = loss
                statistics[f"elem_acc/edge_{key}"] = elem_acc 
                statistics[f"seq_acc/edge_{key}"] = sequence_acc

            loss_total += loss

        statistics["loss/total"] = loss_total
        statistics["acc/total"] = correct_total.mean()

        return loss_total, statistics

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

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="moses")

        parser.add_argument("--num_encoder_layers", type=int, default=6)
        parser.add_argument("--num_decoder_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)
        parser.add_argument("--vq_vocab_size", type=int, default=32)

        parser.add_argument("--lr", type=float, default=1e-6)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    neptune_logger = NeptuneLogger(project="sungsahn0215/molgen")
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