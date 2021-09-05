import os
from pathlib import Path
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from data.tokenize import tokenize, tokenize_with_singlebond   
from data.smilesstate import SmilesState

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ZincNoSingleBondDataset(Dataset):
    raw_dir = "../resource/data/zinc/raw"
    processed_dir = "../resource/data/zinc/nosinglebond"
    def __init__(self, split):
        tokenizer_path = os.path.join(self.processed_dir, "tokenizer.json")
        strings_path = os.path.join(self.processed_dir, f"{split}.txt")
        if not os.path.exists(tokenizer_path) or not os.path.exists(strings_path):
            self.setup()
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.strings = Path(strings_path).read_text(encoding="utf=8").splitlines()
        
    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        string = self.strings[idx]
        smiles = string.replace(" ", "")
        tokens = self.tokenizer.decode(self.tokenizer.encode(string).ids, skip_special_tokens=False).split(" ")
        return SmilesState(tokens).featurize(self.tokenizer)
    
    def setup(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        all_tokens_list = []
        for split in ["train", "valid", "test"]:
            smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
            smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
            tokens_list = self.process_smiles_list(smiles_list)
            tokens_list_path = os.path.join(self.processed_dir, f"{split}.txt")
            Path(tokens_list_path).write_text("\n".join(tokens_list))

            all_tokens_list += tokens_list

        #
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = WordLevelTrainer(vocab_size=40000, special_tokens=["<pad>", "<mask>", "<bos>", "<eos>"])
        tokenizer.train_from_iterator(iter(all_tokens_list), trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>")),],
        )
        tokenizer.save(os.path.join(self.processed_dir, "tokenizer.json"))
        
    def process_smiles_list(self, smiles_list):
        return Parallel(n_jobs=8)(delayed(tokenize)(smiles) for smiles in smiles_list)

class ZincYesSingleBondDataset(ZincNoSingleBondDataset):
    raw_dir = "../resource/data/zinc/raw"
    processed_dir = "../resource/data/zinc/yessinglebond"
    def process_smiles_list(self, smiles_list):
        return Parallel(n_jobs=8)(delayed(tokenize_with_singlebond)(smiles) for smiles in smiles_list)