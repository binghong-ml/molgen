import os
import argparse
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from model import BaseGenerator
from data.dataset import ZincNoSingleBondDataset, ZincYesSingleBondDataset
from data.smilesstate import SmilesState
from util import compute_sequence_cross_entropy, canonicalize

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
    
    def setup_datasets(self, hparams):
        if hparams.dataset_name == "nosinglebond":
            dataset_cls = ZincNoSingleBondDataset
        elif hparams.dataset_name == "yessinglebond":
            dataset_cls = ZincYesSingleBondDataset
            
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.tokenizer = self.train_dataset.tokenizer
        
        self.collate = SmilesState.collate

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            self.tokenizer,
            hparams.num_encoder_layers,
            hparams.emb_size,
            hparams.nhead,
            hparams.dim_feedforward,
            hparams.dropout,
            hparams.use_linedistance,
            hparams.use_distance,
            hparams.use_equality,
            hparams.use_isopen,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        sequences, linedistances, distances, equalities, isopens, _ = batched_data
        
        # decoding
        logits = self.model(sequences, linedistances, distances, isopens, equalities)
        recon_loss = compute_sequence_cross_entropy(logits, sequences)
        loss += recon_loss
        
        statistics["loss/total"] = loss
                
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
        statistics = dict()
        
        with torch.no_grad():
            states = self.model.decode(
                self.hparams.sample_batch_size,
                max_len=120,
                device=self.device
                )
        
        maybe_smiles_list = [state.get_smiles() for state in states]
        smiles_list = []
        for maybe_smiles in maybe_smiles_list:
            smiles = canonicalize(maybe_smiles)
            if smiles is None:
                if self.current_epoch > 0:
                    self.logger.experiment["invalid_smiles"].log(maybe_smiles)
            else:
                smiles_list.append(smiles)
        
        unique_smiles_list = list(set(smiles_list))

        statistics["sample/valid"] = float(len(smiles_list)) / self.hparams.sample_batch_size
        statistics["sample/unique"] = float(len(unique_smiles_list)) / self.hparams.sample_batch_size
        
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="yessinglebond")
        
        parser.add_argument("--num_encoder_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        
        parser.add_argument("--use_linedistance", action="store_true")
        parser.add_argument("--use_distance", action="store_true")
        parser.add_argument("--use_equality", action="store_true")
        parser.add_argument("--use_isopen", action="store_true")

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        
        parser.add_argument("--sample_batch_size", type=int, default=256)
        
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = BaseGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_from_checkpoint(hparams.load_checkpoint_path)
    
    neptune_logger = NeptuneLogger(project="sungsahn0215/molgen", close_after_fit=False)
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run['sys/tags'].add(hparams.tag.split("_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("../resource/checkpoint/", hparams.tag),
        monitor="validation/loss/total",
        mode="min"
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