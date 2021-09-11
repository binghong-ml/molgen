import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from data.util import smiles_to_tsrs, node_feature_names, edge_feature_names

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "../resource/data"

def collate_squares(dicts):
    return 
    
def pad_squares(squares, pad_id):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), pad_id, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares

def collate(data_list):
    node_data_list, edge_data_list = zip(*data_list)
    batched_node_data = {
        key: pad_sequence([d[key] for d in node_data_list], batch_first=True, padding_value=0) 
        for key in node_feature_names
        }
    batched_edge_data = {key: pad_squares([d[key] for d in edge_data_list], 0) for key in edge_feature_names}
    return batched_node_data, batched_edge_data 

class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc/raw"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        node_tsrs, edge_tsrs = smiles_to_tsrs(smiles)
        return node_tsrs, edge_tsrs

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses/raw"
    
class PolymerDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/polymers/raw"
