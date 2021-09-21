from data.smiles import smiles2molgraph
from data.smiles_dataset import tokenize, untokenize
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.target_data2 import Data
from data.smiles2 import molgraph2smiles, smiles2molgraph

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

tokens = []
for smiles_list_dir in ["../resource/data/zinc"]:
    for split in ["test", "train", "valid"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = [
            smiles
            for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
            for smiles in pair.split()
        ]
        pass_smiles_list = []
        nonpass_smiles_list = []
        
        for smiles in tqdm(smiles_list):
            smiles = canonicalize(smiles)
            try:
                recon_smiles = Data.from_smiles(smiles).to_smiles()
                pass_smiles_list.append(smiles)
            except:
                print(smiles)
                nonpass_smiles_list.append(smiles)            
                assert False
        
        pass_smiles_list_path = f"{smiles_list_dir}/{split}_passed.txt"
        Path(pass_smiles_list_path).write_text("\n".join(pass_smiles_list))

        nonpass_smiles_list_path = f"{smiles_list_dir}/{split}_passed.txt"
        Path(nonpass_smiles_list_path).write_text("\n".join(nonpass_smiles_list))
                     
