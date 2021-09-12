import os
from pathlib import Path
from torch.utils.data import Dataset
from data.util import Data

DATA_DIR = "../resource/data"
class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc/raw"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return Data.from_smiles(smiles).featurize()

class ZincAutoEncoderDataset(ZincDataset):
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = Data.from_smiles((smiles))
        return sequence, sequence

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"
    
class MosesAutoEncoderDataset(ZincAutoEncoderDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"

class PolymerDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/polymers/raw"

class LogP04Dataset(Dataset):
    raw_dir = f"{DATA_DIR}/logp04/raw"
    def __init__(self, split):
        self.split = split
        if self.split == "train":
            smiles_list_path = os.path.join(self.raw_dir, "train_pairs.txt")
            smiles_pair_list = [
                pair.split() for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
                ]
            self.src_smiles_list, self.tgt_smiles_list = map(list, zip(*smiles_pair_list))
        else:
            smiles_list_path = os.path.join(self.raw_dir, f"{self.split}.txt")
            self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()

    def __len__(self):
        if self.split == "train":
            return len(self.src_smiles_list)
        else:
            return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.split == "train":
            src_smiles = self.src_smiles_list[idx]
            tgt_smiles = self.tgt_smiles_list[idx]
            return Data.from_smiles(src_smiles).featurize(), Data.from_smiles(tgt_smiles).featurize()
        else:
            smiles = self.smiles_list[idx]
            sequence = Data.from_smiles(smiles).featurize()
            return sequence, sequence

class LogP06Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/logp06/raw"

class DRD2Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/drd2/raw"

class QEDDataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/qed/raw"
