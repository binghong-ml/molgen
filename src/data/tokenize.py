from rdkit import Chem
import enum
import networkx as nx
from networkx.readwrite.gml import Token


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
    tokens = []
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

    return tokens


def tokenize_with_singlebond(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, allBondsExplicit=True)
    return tokenize(smiles)

def get_tokentype(token):
    if token.startswith("[") or token in ORGANIC_ATOMS:
        return TokenType.ATOM
    elif token.startswith("%") or token.isdigit():
        return TokenType.RING_NUM
    elif token in "-=#$:./\\":
        return TokenType.BOND
    elif token == "(":
        return TokenType.BRANCH_START
    elif token == ")":
        return TokenType.BRANCH_END
    elif token in ["<bos>", "<eos>", "<mask>", "<unk>", "<pad>"]:
        return TokenType.SPECIAL
    else:
        raise ValueError(f"Undefined tokentype for {token}.")


def get_bondorder(token):
    return {"-": 1, "=": 2, "#": 3, "$": 4, ":": 1.5, ".": 0, "/": 1, "\\": 1}.get(token)
