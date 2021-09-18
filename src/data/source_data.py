import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy
from data.smiles import TOKEN2ATOMFEAT, TOKEN2BONDFEAT, smiles2molgraph
from util import pad_square

import torch
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "[pad]"
ATOM_TOKENS = [token for token in TOKEN2ATOMFEAT]
BOND_TOKENS = [token for token in TOKEN2BONDFEAT]
TOKENS = [PAD_TOKEN] + ATOM_TOKENS + BOND_TOKENS

TOKEN2ID = {token: idx for idx, token in enumerate(TOKENS)}
ID2TOKEN = {idx: token for idx, token in enumerate(TOKENS)}

MAX_LEN = 250


def get_id(token):
    return TOKEN2ID[token]


def get_ids(tokens):
    return [TOKEN2ID[token] for token in tokens]


class Data:
    def __init__(self, sequence, distance_square):
        self.sequence = sequence
        self.distance_square = distance_square

    @staticmethod
    def from_smiles(smiles):
        molgraph = smiles2molgraph(smiles)
        num_atoms = len(molgraph.nodes())
        atom_tokens = nx.get_node_attributes(molgraph, "token")
        bond_tokens = nx.get_edge_attributes(molgraph, "token")

        tokens = [atom_tokens[node] for node in molgraph.nodes()] + [bond_tokens[edge] for edge in molgraph.edges()]
        sequence = get_ids(tokens)

        mollinegraph = nx.Graph()
        for atom_node in molgraph.nodes():
            mollinegraph.add_node(atom_node)

        for idx, edge in enumerate(molgraph.edges()):
            atom_node0, atom_node1 = edge
            bond_node = num_atoms + idx
            mollinegraph.add_node(bond_node)
            mollinegraph.add_edge(atom_node0, bond_node)
            mollinegraph.add_edge(bond_node, atom_node1)

        distance_square = floyd_warshall_numpy(mollinegraph, weight="weight")
        distance_square[np.isinf(distance_square)] = -1
        distance_square = distance_square.astype(int) + 1
        distance_square[distance_square > MAX_LEN] = MAX_LEN

        return Data(sequence, distance_square)

    def featurize(self):
        return torch.LongTensor(self.sequence), torch.LongTensor(self.distance_square)

    @staticmethod
    def collate(data_list):
        sequences, distance_squares = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        distance_squares = pad_square(distance_squares, padding_value=0)
        return sequences, distance_squares
