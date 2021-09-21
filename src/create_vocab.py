from data.smiles import smiles2molgraph
from data.smiles_dataset import tokenize, untokenize
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.target_data2 import Data
from data.smiles2 import get_bond_order, molgraph2smiles, smiles2molgraph, get_atom_token, get_bond_token

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)

from collections import defaultdict
"""
tokens = []
for smiles_list_dir in ["../resource/data/zinc"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = [
            smiles
            for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
            for smiles in pair.split()
        ]
        token2valence = defaultdict(int)
        for smiles in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            Chem.RemoveStereochemistry(mol)

            atom_aggr_bond_orders = defaultdict(int)
            for bond in mol.GetBonds():
                atom_aggr_bond_orders[bond.GetBeginAtomIdx()] += get_bond_order(get_bond_token(bond))
                atom_aggr_bond_orders[bond.GetEndAtomIdx()] += get_bond_order(get_bond_token(bond))

            for idx, atom in enumerate(mol.GetAtoms()):
                token = get_atom_token(atom)
                token2valence[token] = max(token2valence[token], atom_aggr_bond_orders[idx])

        print(token2valence) 
"""

Data.from_smiles("I").to_smiles()
assert False

for smiles_list_dir in ["../resource/data/zinc"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = [
            smiles
            for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
            for smiles in pair.split()
        ]

        for smiles in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            Chem.RemoveStereochemistry(mol)
            recon_smiles0 = Chem.MolToSmiles(mol, kekuleSmiles=True)
            
            recon_smiles1 = Data.from_smiles(recon_smiles0).to_smiles()
            
            mol = Chem.MolFromSmiles(recon_smiles1)
            Chem.Kekulize(mol)
            Chem.RemoveStereochemistry(mol)
            recon_smiles1 = Chem.MolToSmiles(mol, kekuleSmiles=True)
            
            if canonicalize(recon_smiles0) != canonicalize(recon_smiles1):
                print(smiles)
                print(recon_smiles0)
                print(recon_smiles1)

            
                
        # smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        #for smiles in tqdm(smiles_list):
        #    recon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        #    print(recon_smiles)
        #    assert smiles == recon_smiles

            # data = SourceData.from_smiles(smiles)

            # recon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(data.to_smiles()[0]))
            # if smiles != recon_smiles:
            #    print(smiles)
            #    print(recon_smiles)
