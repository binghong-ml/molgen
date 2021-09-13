from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.util import Data, smiles_to_nx, nx_to_smiles
import networkx as nx
import json

seen_node_feats = set()
seen_edge_feats = set()
for smiles_list_dir in ["../resource/data/moses/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        for smiles in tqdm(smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            
            #G = smiles_to_nx(smiles)
            #node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence = nx_to_sequence(G)
            #G = sequence_to_nx(node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence)
            #recon_smiles = nx_to_smiles(G)
            data = Data.from_smiles(smiles)
            recon_smiles, error = data.to_smiles()

            if smiles != recon_smiles:
                print(smiles)
                print(recon_smiles)
                print(error)

            (
                atom_or_bond_sequence, 
                atomid_sequence,
                bondid_sequence,
                queueid_sequences, 
                adj_square, 
                atom_queue_id_square, 
                bond_queue_id_square
             ) = data.featurize()
            
            print(atom_or_bond_sequence[:10]) 
            print(atomid_sequence[:10])
            print(bondid_sequence[:10])
            print(queueid_sequences[:10])
            print(adj_square[:10, :10]) 
            print(atom_queue_id_square[:10, :10]) 
            print(bond_queue_id_square[:10, :10])
            assert False
            #node_feats = nx.get_node_attributes(G, 'feature')
            #seen_node_feats = seen_node_feats.union(set(node_feats.values()))

            #edge_feats = nx.get_edge_attributes(G, 'feature')
            #seen_edge_feats = seen_edge_feats.union(set(edge_feats.values()))

print(sorted(seen_node_feats))        
print(sorted(seen_edge_feats))