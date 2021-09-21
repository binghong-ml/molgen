from rdkit import Chem
import enum

PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    "#",
    "(",
    ")",
    "-",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "=",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "[C@@H]",
    "[C@@]",
    "[C@H]",
    "[C@]",
    "[CH-]",
    "[CH2-]",
    "[H]",
    "[N+]",
    "[N-]",
    "[NH+]",
    "[NH-]",
    "[NH2+]",
    "[NH3+]",
    "[O+]",
    "[O-]",
    "[OH+]",
    "[P+]",
    "[P@@H]",
    "[P@@]",
    "[P@]",
    "[PH+]",
    "[PH2]",
    "[PH]",
    "[S+]",
    "[S-]",
    "[S@@+]",
    "[S@@]",
    "[S@]",
    "[SH+]",
    "[n+]",
    "[n-]",
    "[nH+]",
    "[nH]",
    "[o+]",
    "[s+]",
    "\\",
    "c",
    "n",
    "o",
    "s",
]
TOKEN2ID = {token: TOKENS.index(token) for token in TOKENS}
ID2TOKEN = {idx: TOKENS[idx] for idx in range(len(TOKENS))}
MAX_LEN = 250


@enum.unique
class TokenType(enum.IntEnum):
    ATOM = 1
    BOND = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    SPECIAL = 6


ORGANIC_ATOMS = "B C N O P S F Cl Br I * b c n o s p".split()


def tokenize(smiles):
    smiles = iter(smiles)
    tokens = ["[bos]"]
    peek = None
    while True:
        char = peek if peek else next(smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            token = char + next(smiles, "") + next(smiles, "")

        elif char in "-=#$:.()%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)

    tokens.append("[eos]")
    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence):
    tokens = [ID2TOKEN[id_] for id_ in sequence]
    if tokens[0] != "[bos]":
        return ""
    elif "[eos]" not in tokens:
        return ""

    tokens = tokens[1 : tokens.index("[eos]")]
    return "".join(tokens)


DATA_DIR = "../resource/data"

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    simple = True

    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        mol = Chem.MolFromSmiles(smiles)
        if self.simple:
            Chem.Kekulize(mol)

        smiles = Chem.MolToSmiles(mol)
        return torch.LongTensor(tokenize(smiles))


class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"
    simple = True


class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = True


class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = False
