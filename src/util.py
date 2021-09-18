from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

import torch
import torch.nn.functional as F


def compute_sequence_accuracy(logits, batched_sequence_data, ignore_index=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == ignore_index] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, ignore_index=0):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)

    return loss


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        return None, str(e)

    if len(smiles) == 0:
        return None, "zero length smiles"

    return smiles, None


def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
