import random
from pathlib import Path

smiles_list_path = "../resource/data/qm9/all.txt"
smiles_list = [pair.split(",")[1] for pair in Path(smiles_list_path).read_text(encoding="utf=8").splitlines()[1:]]

random.shuffle(smiles_list)

train_smiles_list = smiles_list[: int(0.8 * len(smiles_list))]
valid_smiles_list = smiles_list[int(0.8 * len(smiles_list)) : int(0.9 * len(smiles_list))]
test_smiles_list = smiles_list[int(0.9 * len(smiles_list)) :]

Path("../resource/data/qm9/train.txt").write_text("\n".join(train_smiles_list))
Path("../resource/data/qm9/valid.txt").write_text("\n".join(valid_smiles_list))
Path("../resource/data/qm9/test.txt").write_text("\n".join(test_smiles_list))
