from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.util import Data

def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
seen_node_feats = set()
seen_edge_feats = set()

avg_smiles_len = 0.0
avg_singlebond_smiles_len = 0.0
avg_tokens_len = 0.0
for smiles_list_dir in ["../resource/data/moses/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()[:1000]
        for smiles in tqdm(smiles_list):
            #G = smiles_to_nx(smiles)
            #node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence = nx_to_sequence(G)
            #G = sequence_to_nx(node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence)
            #recon_smiles = nx_to_smiles(G)
            avg_smiles_len += float(len(smiles)) / len(smiles_list)
            avg_singlebond_smiles_len += float(len(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), allBondsExplicit=True))) / len(smiles_list)
            data = Data.from_smiles(smiles)
            avg_tokens_len += float(len(data.tokens)) / len(smiles_list)
            print(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), allBondsExplicit=True))
            print("".join(data.tokens[1:-1]))
            
            assert False

        print(avg_smiles_len)
        print(avg_singlebond_smiles_len)
        print(avg_tokens_len)

        assert False