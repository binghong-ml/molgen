from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.source_data import Data as SourceData


def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


for smiles_list_dir in ["../resource/data/zinc/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        for smiles in tqdm(smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            data = SourceData.from_smiles(smiles)

            # recon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(data.to_smiles()[0]))
            # if smiles != recon_smiles:
            #    print(smiles)
            #    print(recon_smiles)
