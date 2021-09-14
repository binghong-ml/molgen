from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.tokenization import sequence2nx, smiles2nx, nx2sequence, nx2smiles
import networkx as nx
import json

def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

seen_node_feats = set()
seen_edge_feats = set()
for smiles_list_dir in ["../resource/data/moses/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        for smiles in tqdm(smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            
            try:
                val_seq, pos_seq = nx2sequence(smiles2nx(smiles))
                recon_smiles = canonicalize(nx2smiles(sequence2nx(val_seq, pos_seq)))
            except:
                print(smiles)
                assert False

            if smiles != recon_smiles:
                print(smiles)
                print(recon_smiles)
                assert False