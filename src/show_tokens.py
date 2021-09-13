from pathlib import Path
from util import canonicalize
from joblib import Parallel, delayed
from tqdm import tqdm

from data.tokenize import tokenize_with_singlebond
from data.data import TargetData

DATA_DIR = "../resource/data"
TOKENIZER_PATH = f"{DATA_DIR}/tokenizer.json"
ALL_PATHS = [
    #f"{DATA_DIR}/zinc/raw/train.txt", 
    #f"{DATA_DIR}/zinc/raw/valid.txt", 
    #f"{DATA_DIR}/zinc/raw/test.txt", 
    f"{DATA_DIR}/moses/raw/train.txt", 
    f"{DATA_DIR}/moses/raw/valid.txt", 
    f"{DATA_DIR}/moses/raw/test.txt",
    #f"{DATA_DIR}/logp04/raw/train_pairs.txt", 
    #f"{DATA_DIR}/logp04/raw/valid.txt", 
    #f"{DATA_DIR}/logp04/raw/test.txt",
    #f"{DATA_DIR}/logp06/raw/train_pairs.txt", 
    #f"{DATA_DIR}/logp06/raw/valid.txt", 
    #f"{DATA_DIR}/logp06/raw/test.txt", 
    #f"{DATA_DIR}/drd2/raw/train_pairs.txt", 
    #f"{DATA_DIR}/drd2/raw/valid.txt", 
    #f"{DATA_DIR}/drd2/raw/test.txt",
    #f"{DATA_DIR}/qed/raw/train_pairs.txt", 
    #f"{DATA_DIR}/qed/raw/valid.txt", 
    #f"{DATA_DIR}/qed/raw/test.txt",
    ]

all_tokens_list = set()
for smiles_list_path in tqdm(ALL_PATHS):
    smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    smiles_list = [smiles for elem in smiles_list for smiles in elem.split(", ")]
    
    for smiles in tqdm(smiles_list):
        recon_smiles = TargetData.from_smiles(smiles).to_smiles()
        if recon_smiles != canonicalize(smiles):
            print()
            print()
            print(smiles, recon_smiles)
            assert False

    #tokens_list = Parallel(n_jobs=8)(delayed(tokenize_with_singlebond)(smiles) for smiles in smiles_list)
    #all_tokens_list = all_tokens_list.union(set([token for tokens in tokens_list for token in tokens]))

#print(sorted(all_tokens_list))
