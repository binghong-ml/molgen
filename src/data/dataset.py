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
TOKENIZER_PATH = f"{DATA_DIR}/tokenizer.json"
ALL_PATHS = [
    f"{DATA_DIR}/zinc/raw/train.txt", 
    f"{DATA_DIR}/zinc/raw/valid.txt", 
    f"{DATA_DIR}/zinc/raw/test.txt", 
    f"{DATA_DIR}/moses/raw/train.txt", 
    f"{DATA_DIR}/moses/raw/valid.txt", 
    f"{DATA_DIR}/moses/raw/test.txt",
    f"{DATA_DIR}/logp04/raw/train_pairs.txt", 
    f"{DATA_DIR}/logp04/raw/valid.txt", 
    f"{DATA_DIR}/logp04/raw/test.txt",
    f"{DATA_DIR}/logp06/raw/train_pairs.txt", 
    f"{DATA_DIR}/logp06/raw/valid.txt", 
    f"{DATA_DIR}/logp06/raw/test.txt", 
    f"{DATA_DIR}/drd2/raw/train_pairs.txt", 
    f"{DATA_DIR}/drd2/raw/valid.txt", 
    f"{DATA_DIR}/drd2/raw/test.txt",
    f"{DATA_DIR}/qed/raw/train_pairs.txt", 
    f"{DATA_DIR}/qed/raw/valid.txt", 
    f"{DATA_DIR}/qed/raw/test.txt",
    ]

def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        setup_tokenizer()

    return Tokenizer.from_file(TOKENIZER_PATH)


def setup_tokenizer():
    all_smiles_list = []
    for smiles_list_path in ALL_PATHS:
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        smiles_list = [smiles for elem in smiles_list for smiles in elem.split(", ")]
        all_smiles_list += smiles_list
            
    all_tokens_list = Parallel(n_jobs=8)(delayed(tokenize_with_singlebond)(smiles) for smiles in all_smiles_list)

    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = WordLevelTrainer(vocab_size=40000, special_tokens=["<pad>", "<mask>", "<bos>", "<eos>"])
    tokenizer.train_from_iterator(iter(all_tokens_list), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>")),],
    )
    tokenizer.save(TOKENIZER_PATH)

class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc/raw"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.tokenizer = load_tokenizer()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        string = self.smiles2string(smiles)
        tokens = self.tokenizer.decode(self.tokenizer.encode(string).ids, skip_special_tokens=False).split(" ")
        return TargetData(tokens).featurize(self.tokenizer)

    def smiles2string(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, allBondsExplicit=True)
        return tokenize(smiles)

class ZincAutoEncoderDataset(ZincDataset):
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        string = self.smiles2string(smiles)
        src_tokens = self.tokenizer.decode(self.tokenizer.encode(string).ids, skip_special_tokens=False).split(" ")
        tgt_tokens = self.tokenizer.decode(self.tokenizer.encode(string).ids, skip_special_tokens=False).split(" ")
        
        return SourceData(src_tokens).featurize(self.tokenizer), TargetData(tgt_tokens).featurize(self.tokenizer)

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"
    
class MosesAutoEncoderDataset(ZincAutoEncoderDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"

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
