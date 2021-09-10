from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.util import nx_to_tsrs, tsrs_to_nx, smiles_to_nx, nx_to_smiles

smiles_list_path = "../resource/data/zinc/raw/train.txt"
smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
for smiles in tqdm(smiles_list):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    
    G = smiles_to_nx(smiles)
    recon_smiles = nx_to_smiles(G)
    if smiles != recon_smiles:
        print(1)
        print(smiles)
        print(recon_smiles)

    node_tsrs, edge_tsrs = nx_to_tsrs(G)
    G = tsrs_to_nx(node_tsrs, edge_tsrs)
    recon_smiles = nx_to_smiles(G)
    if smiles != recon_smiles:
        print(2)
        print(smiles)
        print(recon_smiles)
