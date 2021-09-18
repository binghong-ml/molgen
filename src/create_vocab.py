from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.util import Data


def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


seen_node_feats = set()
seen_edge_feats = set()

avg_token_len = 0.0
for smiles_list_dir in ["../resource/data/zinc/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        for smiles in smiles_list[:5000]:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            # G = smiles_to_nx(smiles)
            # node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence = nx_to_sequence(G)
            # G = sequence_to_nx(node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence)
            # recon_smiles = nx_to_smiles(G)
            data = Data.from_smiles(smiles)
            recon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(data.to_smiles()[0]))
            if smiles != recon_smiles:
                print(smiles)
                print(recon_smiles)

            max_token_len = max(max_token_len, len(data.tokens))

        print(max_token_len)
        recon_smiles, error = data.to_smiles()
        if error is not None:
            print(error)
            assert False

        if canonicalize(recon_smiles) != canonicalize(smiles):
            print(smiles)
            print(recon_smiles)

        # recon_smiles = data.to_smiles()

        # if smiles != recon_smiles:
        #    assert False
