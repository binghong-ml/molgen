import argparse
from pathlib import Path
import moses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_list_path", type=str, default="")
    parser.add_argument("--train_smiles_list_path", type=str, default="")
    parser.add_argument("--test_smiles_list_path", type=str, default="")
    hparams = parser.parse_args()

    smiles_list = Path(hparams.smiles_list_path).read_text(encoding="utf-8").splitlines()
    train_smiles_list = Path(hparams.train_smiles_list_path).read_text(encoding="utf-8").splitlines()
    test_smiles_list = Path(hparams.test_smiles_list_path).read_text(encoding="utf-8").splitlines()
    metrics = moses.get_all_metrics(smiles_list, n_jobs=8, device="cuda:0")

    print(metrics)
