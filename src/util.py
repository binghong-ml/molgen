from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

import torch
import torch.nn.functional as F


def compute_sequence_accuracy(logits, batched_sequence_data, pad_id=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == pad_id] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, pad_id=0):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id,)

    return loss


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None

    if len(smiles) == 0:
        return None

    return smiles
