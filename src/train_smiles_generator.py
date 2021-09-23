import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from moses.utils import disable_rdkit_log, enable_rdkit_log

from model.smiles_generator import SmilesGenerator
from train_generator import BaseGeneratorLightningModule
from data.smiles_dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset, untokenize
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize


class SmilesGeneratorLightningModule(BaseGeneratorLightningModule):
    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = SmilesGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
        )

    ### 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data, ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return loss, statistics

    # 
    def sample(self, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            results.extend(untokenize(sequence) for sequence in sequences.tolist())

        disable_rdkit_log()
        smiles_list = list(map(canonicalize, results))
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
    SmilesGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = SmilesGeneratorLightningModule(hparams)
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