import os
import argparse
from pathlib import Path
from joblib.parallel import delayed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from model.vae import VariationalAutoEncoder
from data.dataset import ZincDataset, MosesDataset, PolymerDataset, collate
from data.util import EDGE_TARGET_NAMES, NODE_TARGET_NAMES, tsrs_to_smiles

from rdkit import Chem
import moses
from joblib import Parallel, delayed


class VariationalAutoEncoderLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(VariationalAutoEncoderLightningModule, self).__init__()
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
        self.collate = collate

        self.train_dataset.smiles_list = self.train_dataset.smiles_list
        self.val_dataset.smiles_list = self.val_dataset.smiles_list


        self.train_smiles_set = set(self.train_dataset.smiles_list[:100000])
        get_num_nodes = lambda smiles: len(Chem.MolFromSmiles(smiles).GetAtoms())
        self.num_nodes_list = Parallel(n_jobs=hparams.num_workers)(
            delayed(get_num_nodes)(smiles) for smiles in self.train_smiles_set
            )

    def setup_model(self, hparams):
        self.model = VariationalAutoEncoder(
            num_encoder_layers= hparams.num_encoder_layers,
            num_decoder_layers= hparams.num_decoder_layers,
            emb_size= hparams.emb_size,
            nhead= hparams.nhead,
            dim_feedforward= hparams.dim_feedforward,
            logit_hidden_dim= hparams.logit_hidden_dim,
            code_dim= hparams.code_dim,
            num_nodes_list= self.num_nodes_list,
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
        batched_node_data, batched_edge_data = batched_data
        statistics = self.model.step(batched_node_data, batched_edge_data)
        
        loss = 0.0
        for key in NODE_TARGET_NAMES + EDGE_TARGET_NAMES:
            loss += statistics[f"loss/{key}"]

        loss += self.hparams.reg_coef * statistics[f"loss/reg"]
        statistics["loss/total"] = loss

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
        statistics = dict()
        smiles_list = self.sample(self.hparams.num_samples)

        smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        statistics["sample/valid"] = float(len(smiles_list)) / self.hparams.num_samples
        
        unique_smiles_set = set(smiles_list)
        statistics["sample/unique"] = float(len(unique_smiles_set)) / self.hparams.num_samples
        
        novel_smiles_set = unique_smiles_set - self.train_smiles_set
        statistics["sample/novel"] = float(len(novel_smiles_set)) / self.hparams.num_samples

        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    def sample(self, num_samples):
        smiles_list = []
        while len(smiles_list) < num_samples: 
            cur_num_samples = min(num_samples-len(smiles_list), self.hparams.sample_batch_size)
            cur_smiles_list = self._sample(cur_num_samples)
            smiles_list = smiles_list + cur_smiles_list

        return smiles_list

    def _sample(self, num_samples):
        node_ids, edge_ids = self.model.sample(num_samples, self.device)
        node_ids = {key: node_ids[key].cpu() for key in node_ids}
        edge_ids = {key: edge_ids[key].cpu() for key in edge_ids}

        smiles_list = []
        for idx in range(num_samples):
            node_tsrs = {key: node_ids[key][idx] for key in NODE_TARGET_NAMES}
            edge_tsrs = {key: edge_ids[key][idx] for key in EDGE_TARGET_NAMES}
            if idx == 0:
                print(edge_tsrs)

            smiles = tsrs_to_smiles(node_tsrs, edge_tsrs)
            smiles_list.append(smiles)

        return smiles_list

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="zinc")

        parser.add_argument("--num_encoder_layers", type=int, default=6)
        parser.add_argument("--num_decoder_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--logit_hidden_dim", type=int, default=1024)
        parser.add_argument("--code_dim", type=int, default=64)
        parser.add_argument("--reg_coef", type=float, default=1e-1)
        
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--num_samples", type=int, default=256)
        parser.add_argument("--sample_batch_size", type=int, default=256)
        

        return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    VariationalAutoEncoderLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="../resource/checkpoint/default")
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = VariationalAutoEncoderLightningModule(hparams)
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