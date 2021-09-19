import os
from pathlib import Path
from torch.utils.data import Dataset
from data.target_data import Data as TargetData
from data.source_data import Data as SourceData

DATA_DIR = "../resource/data"


class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    simple = True

    def __init__(self, split, randomize_dfs):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.randomize_dfs = randomize_dfs
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles, simple=self.simple, randomize_dfs=self.randomize_dfs).featurize()


class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"
    simple = True


class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = True


class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = False


class LogP04Dataset(Dataset):
    raw_dir = f"{DATA_DIR}/logp04"
    simple = False

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
            return (
                SourceData.from_smiles(src_smiles).featurize(),
                TargetData.from_smiles(tgt_smiles).featurize(),
            )
        else:
            smiles = self.smiles_list[idx]
            return SourceData.from_smiles(smiles).featurize(), smiles


class LogP06Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/logp06"
    simple = False


class DRD2Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/drd2"
    simple = False


class QEDDataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/qed"
    simple = False
