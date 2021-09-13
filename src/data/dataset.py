import os
from pathlib import Path
from joblib import Parallel, delayed
from rdkit import Chem

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from data.tokenize import tokenize, tokenize_with_singlebond
from data.data import TargetData, SourceData

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "../resource/data"
class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc/raw"
    def __init__(self, split, randomize):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.randomize = randomize
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles, randomize=self.randomize).featurize()

class ZincAutoEncoderDataset(ZincDataset):
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        return SourceData.from_smiles(smiles).featurize(), TargetData.from_smiles(smiles).featurize(), smiles

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"
    
class MosesAutoEncoderDataset(ZincAutoEncoderDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"

"""
class PolymerDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/polymers/raw"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.tokenizer = load_polymer_tokenizer()

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

        self.tokenizer = load_tokenizer()

    def __len__(self):
        if self.split == "train":
            return len(self.src_smiles_list)
        else:
            return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.split == "train":
            src_smiles = self.src_smiles_list[idx]
            tgt_smiles = self.tgt_smiles_list[idx]
            
            src_string = self.smiles2string(src_smiles)
            tgt_string = self.smiles2string(tgt_smiles)

            src_tokens = self.tokenizer.decode(
                self.tokenizer.encode(src_string).ids, skip_special_tokens=False
                ).split(" ")
            tgt_tokens = self.tokenizer.decode(
                self.tokenizer.encode(tgt_string).ids, skip_special_tokens=False
                ).split(" ")
            
            return SourceData(src_tokens).featurize(self.tokenizer), TargetData(tgt_tokens).featurize(self.tokenizer)

        else:
            smiles = self.smiles_list[idx]
            string = self.smiles2string(smiles)
            tokens = self.tokenizer.decode(self.tokenizer.encode(string).ids, skip_special_tokens=False).split(" ")
            return SourceData(tokens).featurize(self.tokenizer), smiles

    def smiles2string(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, allBondsExplicit=True)
        return tokenize(smiles)

class LogP06Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/logp06/raw"

class DRD2Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/drd2/raw"

class QEDDataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/qed/raw"

"""