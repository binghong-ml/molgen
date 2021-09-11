from hashlib import new
from model.setgenerator import SetGenerator
import os
import argparse
from pathlib import Path

import torch
from torch._C import device
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import moses

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, PolymerDataset, collate
from data.util import node_feature_names, edge_feature_names, nx_to_smiles, tsrs_to_nx

from tqdm import tqdm

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_model(hparams)
        self.setup_datasets(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        dataset_cls = {"zinc": ZincDataset, "moses": MosesDataset, "polymer": PolymerDataset}.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)

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

        self.plug_model = SetGenerator(
            hparams.vq_vocab_size + 1, # + 1 for padding
            hparams.plug_num_encoder_layers,
            hparams.plug_num_decoder_layers,
            hparams.plug_emb_size,
            hparams.plug_nhead,
            hparams.plug_dim_feedforward,
            hparams.plug_dropout,
            hparams.plug_latent_dim
        )

    def setup_prior_datasets(self):
        dataset_cls = {"zinc": ZincDataset, "moses": MosesDataset, "polymer": PolymerDataset}.get(hparams.dataset_name)        
        device = torch.device(0)
        self.model.to(device)
        self.model.eval()
        
        prior_datasets = []
        num_atoms = []
        for split in ["train", "valid"]:
            dataset = dataset_cls(split)
            dataset.smiles_list = dataset.smiles_list[:self.hparams.batch_size * 50]

            loader = DataLoader(
                dataset, 
                batch_size = self.hparams.batch_size, 
                shuffle=False, 
                collate_fn=collate, 
                num_workers=self.hparams.num_workers
            )
            prior_dataset = []
            for batched_data in tqdm(loader):
                batched_node_data, batched_edge_data = batched_data
                batched_node_data = {key: tsr.to(device) for key, tsr in batched_node_data.items()}
                batched_edge_data = {key: tsr.to(device) for key, tsr in batched_edge_data.items()}
                with torch.no_grad():
                    inds, key_padding_mask = self.model.encode(batched_node_data, batched_edge_data)

                inds = inds.cpu() + 1 # add 1 for padding
                key_padding_mask = key_padding_mask.cpu()
                for idx in range(inds.size(0)):
                    prior_dataset.append(inds[idx][~key_padding_mask[idx]])
                
                num_atoms += (~key_padding_mask).sum(dim=1).tolist()
                    
            prior_datasets.append(prior_dataset)
        
        self.train_prior_dataset, self.val_prior_dataset = prior_datasets

        self.plug_model.update_num_latent_prior(num_atoms)

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_prior_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_prior_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        logits, loss_kl = self.plug_model(batched_data)
        flat_logits = logits.view(-1, self.hparams.vq_vocab_size + 1)
        targets = batched_data 
        flat_targets = targets.view(-1)
        loss_recon = F.cross_entropy(flat_logits, flat_targets, ignore_index=0).mean()

        preds = torch.argmax(logits, -1)
        correct = preds == targets
        recon_elem_acc = correct.float().mean()
        recon_total_acc = correct.all(dim=1).float().mean()

        loss_total = loss_recon + self.hparams.plug_kl_coef * loss_kl

        statistics["loss/total"] = loss_total
        statistics["loss/recon"] = loss_recon
        statistics["loss/kl"] = loss_kl
        statistics["acc/total"] = recon_total_acc
        statistics["acc/elem"] = recon_elem_acc

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

    def validation_epoch_end(self, outputs):
        smiles_list = self.sample(self.hparams.num_samples)

        smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        unique_smiles_set = set(smiles_list)
        novel_smiles_set = unique_smiles_set - self.train_smiles_set

        statistics = dict()
        statistics["sample/valid"] = float(len(smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = (
            float(len(unique_smiles_set)) / self.hparams.num_samples if len(smiles_list) > 0 else 0.0
        )
        statistics["sample/novel"] = (
            float(len(novel_smiles_set)) / self.hparams.num_samples if len(smiles_list) > 0 else 0.0
        )

        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True)

    def sample(self, num_samples):
        smiles_list = self._sample(num_samples)
        return smiles_list

    def _sample(self, num_samples):
        inds, key_padding_mask = self.plug_model.sample(num_samples, self.device)
        node_ids, edge_ids = self.model.decode(inds, key_padding_mask)

        node_ids = {key: node_ids[key].cpu() for key in node_ids}
        edge_ids = {key: edge_ids[key].cpu() for key in edge_ids}

        smiles_list = []
        for idx in range(num_samples):
            num_atoms = (~key_padding_mask[idx]).sum()

            node_tsrs = {key: node_ids[key][idx, :num_atoms] for key in node_feature_names}
            edge_tsrs = {key: edge_ids[key][idx, :num_atoms, :num_atoms] for key in edge_feature_names}            
            
            try:
                print(edge_tsrs['bond_type'])
                edge_tsrs['adj'] = (edge_tsrs['bond_type'] > 0).long()
                G = tsrs_to_nx(node_tsrs, edge_tsrs)
                smiles = nx_to_smiles(G)
                smiles_list.append(smiles)

            except Exception as inst:
                print(inst)
                smiles_list.append(None)
        
        print(smiles_list)

        return smiles_list

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

        parser.add_argument("--plug_num_encoder_layers", type=int, default=6)
        parser.add_argument("--plug_num_decoder_layers", type=int, default=6)
        parser.add_argument("--plug_emb_size", type=int, default=1024)
        parser.add_argument("--plug_nhead", type=int, default=8)
        parser.add_argument("--plug_dim_feedforward", type=int, default=2048)
        parser.add_argument("--plug_dropout", type=float, default=0.1)
        parser.add_argument("--plug_latent_dim", type=int, default=4)
        parser.add_argument("--plug_kl_coef", type=float, default=0.1)
        

        parser.add_argument("--lr", type=float, default=1e-6)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--num_samples", type=int, default=256)

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
        state_dict = model.state_dict()
        new_state_dict = torch.load(hparams.load_checkpoint_path)["state_dict"]
        for key in new_state_dict:
            if key in state_dict:
                state_dict[key] = new_state_dict[key]

        model.load_state_dict(state_dict)

    model.setup_prior_datasets()

    #neptune_logger = NeptuneLogger(project="sungsahn0215/molgen")
    #neptune_logger.run["params"] = vars(hparams)
    #neptune_logger.run["sys/tags"].add(hparams.tag.split("_"))
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath=os.path.join("../resource/checkpoint/", hparams.tag), monitor="validation/loss/total", mode="min",
    #)
    trainer = pl.Trainer(
        gpus=1,
        #logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        #callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)