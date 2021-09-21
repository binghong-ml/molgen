# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import random
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy
import numpy as np
import networkx as nx
import torch

from copy import copy, deepcopy

from torch.nn.utils.rnn import pad_sequence

from data.smiles2 import (
    TOKEN2ATOMFEAT,
    TOKEN2BONDFEAT,
    get_max_valence,
    molgraph2smiles,
    smiles2molgraph,
    get_bond_order,
)
from data.dfs import dfs_successors

from util import pad_square
from collections import defaultdict


BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
PAD_TOKEN = "[pad]"
MASK_TOKEN = "[mask]"
SPECIAL_TOKENS = ["[pad]", "[mask]", "[bos]", "[eos]"]
ATOM_TOKENS = [token for token in TOKEN2ATOMFEAT]
BOND_TOKENS = [token for token in TOKEN2BONDFEAT]
BRANCH_START_TOKEN = "("
BRANCH_END_TOKEN = ")"
BRANCH_TOKENS = [BRANCH_START_TOKEN, BRANCH_END_TOKEN]

RING_START_TOKEN = "[bor]"
POSSIBLE_RING_IDXS = 100
RING_END_TOKENS = [f"[eor{idx}]" for idx in range(POSSIBLE_RING_IDXS)]

TOKENS = SPECIAL_TOKENS + BRANCH_TOKENS + ATOM_TOKENS + BOND_TOKENS + [RING_START_TOKEN] + RING_END_TOKENS

RING_ID_START = len(TOKENS) - len(RING_END_TOKENS)
RING_ID_END = len(TOKENS)

TOKEN2ID = {token: idx for idx, token in enumerate(TOKENS)}
ID2TOKEN = {idx: token for idx, token in enumerate(TOKENS)}


def get_id(token):
    return TOKEN2ID[token]


def get_ids(tokens):
    return [TOKEN2ID[token] for token in tokens]


def get_token(id):
    return TOKENS[id]


def get_ring_end_token(idx):
    return f"[eor{idx}]"


def get_ring_end_idx(token):
    return RING_END_TOKENS.index(token)


MAX_LEN = 250


class Data:
    def __init__(self):
        self.sequence = []
        self.tokens = []
        self.node_to_token = dict()
        self.node_to_valence = dict()

        #
        self._node_offset = -1
        self._ring_offset = -1

        #
        self.pointer_node_traj = []

        #
        self.up_loc_square = -np.ones((MAX_LEN, MAX_LEN), dtype=int)
        self.down_loc_square = -np.ones((MAX_LEN, MAX_LEN), dtype=int)

        #
        self.branch_start_nodes = []
        self.ring_to_nodes = defaultdict(list)

        #
        self.started = False
        self.ended = False
        self.error = None

        #
        self.valence_mask_traj = []
        self.graph_mask_traj = []

        #
        self.update(get_id(BOS_TOKEN))

    def __len__(self):
        return len(self.G.nodes())

    def update(self, id):
        token = get_token(id)
        if len(self.graph_mask_traj) == 0:
            if token != BOS_TOKEN:
                self.ended = True
                self.error = "add token without bos"
                return

        elif self.graph_mask_traj[-1][id]:
            self.ended = True
            self.error = "caught by graph mask"
            return

        elif self.valence_mask_traj[-1][id]:
            self.ended = True
            self.error = "caught by valency mask"
            return

        self.sequence.append(id)
        self.tokens.append(token)

        if token in (ATOM_TOKENS + BOND_TOKENS):
            self._node_offset += 1
            new_node = copy(self._node_offset)

            self.node_to_token[new_node] = token

            self.up_loc_square[new_node, new_node] = 0
            self.down_loc_square[new_node, new_node] = 0
            if new_node > 0:
                pointer_node = self.pointer_node_traj[-1]
                self.up_loc_square[new_node, :new_node] = self.up_loc_square[pointer_node, :new_node] + 1
                self.down_loc_square[new_node, :new_node] = self.down_loc_square[pointer_node, :new_node]

                self.up_loc_square[:new_node, new_node] = self.up_loc_square[:new_node, pointer_node]
                self.down_loc_square[:new_node, new_node] = self.down_loc_square[:new_node, pointer_node] + 1

            self.pointer_node_traj.append(new_node)

        elif token == BRANCH_START_TOKEN:
            pointer_node = self.pointer_node_traj[-1]
            self.branch_start_nodes.append(pointer_node)
            self.pointer_node_traj.append(pointer_node)

        elif token == BRANCH_END_TOKEN:
            pointer_node = self.branch_start_nodes.pop()
            self.pointer_node_traj.append(pointer_node)

        elif token == RING_START_TOKEN:
            pointer_node = self.pointer_node_traj[-1]
            self._ring_offset += 1
            new_ring = copy(self._ring_offset)
            self.ring_to_nodes[new_ring].append(pointer_node)
            self.pointer_node_traj.append(pointer_node)

        elif token in RING_END_TOKENS:
            pointer_node = self.pointer_node_traj[-1]
            ring = get_ring_end_idx(token)
            self.ring_to_nodes[ring].append(pointer_node)
            self.pointer_node_traj.append(pointer_node)

        elif token == BOS_TOKEN:
            self.started = True

        elif token == EOS_TOKEN:
            self.ended = True

        # compute graph mask
        if token in ATOM_TOKENS:
            allowed_next_tokens = BOND_TOKENS + [BRANCH_START_TOKEN, RING_START_TOKEN]
            if not self.all_branch_closed():
                allowed_next_tokens.append(BRANCH_END_TOKEN)
            else:
                allowed_next_tokens.append(EOS_TOKEN)

        elif token in BOND_TOKENS:
            allowed_next_tokens = deepcopy(ATOM_TOKENS)
            for ring in self.ring_to_nodes:
                if len(self.ring_to_nodes[ring]) == 1 and self.ring_to_nodes[ring][0] != pointer_node:
                    allowed_next_tokens.append(get_ring_end_token(ring))

        elif token == BRANCH_START_TOKEN:
            allowed_next_tokens = BOND_TOKENS

        elif token == BRANCH_END_TOKEN:
            allowed_next_tokens = [BRANCH_START_TOKEN]
            if not self.all_branch_closed():
                allowed_next_tokens.append(BRANCH_END_TOKEN)
            else:
                allowed_next_tokens.append(EOS_TOKEN)

        elif token == RING_START_TOKEN:
            allowed_next_tokens = BOND_TOKENS + [BRANCH_START_TOKEN, RING_START_TOKEN]
            if not self.all_branch_closed():
                allowed_next_tokens.append(BRANCH_END_TOKEN)
            elif self.all_branch_closed():
                allowed_next_tokens.append(EOS_TOKEN)

        elif token in RING_END_TOKENS:
            allowed_next_tokens = []
            if not self.all_branch_closed():
                allowed_next_tokens.append(BRANCH_END_TOKEN)
            else:
                allowed_next_tokens.append(EOS_TOKEN)

        elif token == BOS_TOKEN:
            allowed_next_tokens = ATOM_TOKENS

        elif token == EOS_TOKEN:
            allowed_next_tokens = []

        graph_mask = np.ones(len(TOKENS), dtype=bool)
        graph_mask[get_ids(allowed_next_tokens)] = False
        self.graph_mask_traj.append(graph_mask)

        # compute valency mask
        valence_mask = np.zeros(len(TOKENS), dtype=bool)
        if token in ATOM_TOKENS:
            valence = get_max_valence(token)
            if new_node > 0:
                valence -= get_bond_order(self.node_to_token[pointer_node])

            self.node_to_valence[new_node] = valence

            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[get_ids(forbidden_bond_tokens)] = True

            if valence < 2:
                valence_mask[get_id(RING_START_TOKEN)] = True
                valence_mask[get_id(BRANCH_START_TOKEN)] = True

        elif token in BOND_TOKENS:
            bond_order = get_bond_order(token)
            self.node_to_valence[pointer_node] -= bond_order
            
            forbidden_atom_tokens = [token_ for token_ in ATOM_TOKENS if get_max_valence(token_) < bond_order]
            
            forbidden_rings = [
                get_ring_end_token(ring)
                for ring in self.ring_to_nodes
                if self.node_to_valence[self.ring_to_nodes[ring][0]] < (bond_order - 1)
            ]

            valence_mask[get_ids(forbidden_atom_tokens)] = True
            valence_mask[get_ids(forbidden_rings)] = True

        elif token == BRANCH_START_TOKEN:
            valence = self.node_to_valence[pointer_node]
            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[get_ids(forbidden_bond_tokens)] = True

        elif token == BRANCH_END_TOKEN:
            if self.node_to_valence[pointer_node] == 0:
                valence_mask[get_id(BRANCH_START_TOKEN)] = True

        elif token == RING_START_TOKEN:
            self.node_to_valence[pointer_node] -= 1

            valence = self.node_to_valence[pointer_node]
            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[get_ids(forbidden_bond_tokens)] = True
            if valence < 2:
                valence_mask[get_id(RING_START_TOKEN)] = True
                valence_mask[get_id(BRANCH_START_TOKEN)] = True

        elif token in RING_END_TOKENS:
            prev_bond_order = get_bond_order(self.node_to_token[pointer_node])
            ring = get_ring_end_idx(token)
            self.node_to_valence[self.ring_to_nodes[ring][0]] -= prev_bond_order - 1

        self.valence_mask_traj.append(valence_mask)

    def all_branch_closed(self):
        return len(self.branch_start_nodes) == 0

    def all_ring_closed(self):
        return all([(len(self.ring_to_nodes[ring]) == 2) for ring in self.ring_to_nodes])

    def to_smiles(self):
        num_nodes = self._node_offset + 1
        up_loc_square = self.up_loc_square[:num_nodes, :num_nodes]
        down_loc_square = self.down_loc_square[:num_nodes, :num_nodes]

        node0s, node1s = ((up_loc_square + down_loc_square) == 1).nonzero()
        node0s, node1s = node0s[node0s < node1s], node1s[node0s < node1s]

        mollinegraph = nx.Graph()
        mollinegraph.add_nodes_from(list(range(num_nodes)))
        mollinegraph.add_edges_from(zip(node0s, node1s))
        for _, ring_nodes in self.ring_to_nodes.items():
            if len(ring_nodes) == 2:
                node0, node1 = ring_nodes
                mollinegraph.add_edge(node0, node1)

        molgraph = nx.Graph()
        for node in mollinegraph.nodes():
            token = self.node_to_token[node]
            if token in ATOM_TOKENS:
                molgraph.add_node(node, token=token)
            elif token in BOND_TOKENS:
                try:
                    node0, node1 = mollinegraph.neighbors(node)
                except:
                    print(self.tokens)
                    assert False

                molgraph.add_edge(node0, node1, token=token)

        smiles = molgraph2smiles(molgraph)

        return smiles

    @staticmethod
    def from_smiles(smiles, randomize=False):
        molgraph = smiles2molgraph(smiles)
        atom_tokens = nx.get_node_attributes(molgraph, "token")
        bond_tokens = nx.get_edge_attributes(molgraph, "token")
        bond_tokens.update({(node1, node0): val for (node0, node1), val in bond_tokens.items()})

        tokens = nx.get_node_attributes(molgraph, "token")

        mollinegraph = nx.Graph()
        for node in molgraph.nodes:
            mollinegraph.add_node(node)

        for edge in molgraph.edges:
            u, v = edge
            mollinegraph.add_node(edge)
            mollinegraph.add_edge(u, edge)
            mollinegraph.add_edge(v, edge)

        def keyfunc(idx):
            return (molgraph.degree(idx), molgraph.nodes[idx].get("token")[0] == 6, idx)

        start = min(molgraph.nodes, key=keyfunc)
        successors = dfs_successors(mollinegraph, source=start, randomize_neighbors=randomize)
        predecessors = dict()
        for node0 in successors:
            for node1 in successors[node0]:
                predecessors[node1] = node0

        #
        edges = set()
        for n_idx, n_jdxs in successors.items():
            for n_jdx in n_jdxs:
                edges.add((n_idx, n_jdx))
                edges.add((n_jdx, n_idx))

        ring_edges = [edge for edge in mollinegraph.edges if tuple(edge) not in edges]

        node_to_ring_idx = defaultdict(list)
        for ring_idx, (atom_node, bond_node) in enumerate(ring_edges):
            node_to_ring_idx[atom_node].append(ring_idx)
            node_to_ring_idx[bond_node].append(ring_idx)

        tokens = []
        to_visit = [start]
        seen_ring_idxs = []
        while to_visit:
            current = to_visit.pop()
            if current in [BRANCH_START_TOKEN, BRANCH_END_TOKEN]:
                tokens.append(current)

            elif current in atom_tokens:
                tokens.append(atom_tokens[current])

            elif current in bond_tokens:
                tokens.append(bond_tokens[current])

            else:
                assert False

            if current in node_to_ring_idx:
                for ring_idx in node_to_ring_idx[current]:
                    if ring_idx not in seen_ring_idxs:
                        tokens.append(RING_START_TOKEN)
                        seen_ring_idxs.append(ring_idx)
                    else:
                        tokens.append(get_ring_end_token(seen_ring_idxs.index(ring_idx)))

            next_nodes = successors.get(current, [])
            if len(next_nodes) == 1:
                to_visit.append(next_nodes[0])

            elif len(next_nodes) > 1:
                for next_node in reversed(next_nodes):
                    to_visit.append(BRANCH_END_TOKEN)
                    to_visit.append(next_node)
                    to_visit.append(BRANCH_START_TOKEN)

        data = Data()
        for token in tokens:
            data.update(get_id(token))
            if data.error is not None:
                print(data.error)

        data.update(get_id(EOS_TOKEN))

        return data

    def featurize(self):
        #
        sequence_len = len(self.sequence)
        sequence = torch.LongTensor(np.array(self.sequence))
        
        ring_start_mask = (sequence == get_id(RING_START_TOKEN))
        count_sequence = ring_start_mask.long().cumsum(dim=0)
        count_sequence = count_sequence.masked_fill(ring_start_mask, 0)

        graph_mask_sequence = torch.tensor(np.array(self.graph_mask_traj), dtype=torch.bool)
        valency_mask_sequence = torch.tensor(np.array(self.valence_mask_traj), dtype=torch.bool)

        #
        linear_loc_square = (
            torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1)) + 1
        )

        #
        pad_right = 1 if self.ended else 0

        up_loc_square = self.up_loc_square[self.pointer_node_traj][:, self.pointer_node_traj]
        up_loc_square = np.pad(up_loc_square + 1, (1, pad_right), "constant")
        up_loc_square = torch.LongTensor(up_loc_square)

        down_loc_square = self.down_loc_square[self.pointer_node_traj][:, self.pointer_node_traj]
        down_loc_square = np.pad(down_loc_square + 1, (1, pad_right), "constant")
        down_loc_square = torch.LongTensor(down_loc_square)

        return sequence, count_sequence, graph_mask_sequence, valency_mask_sequence, linear_loc_square, up_loc_square, down_loc_square

    @staticmethod
    def collate(data_list):
        (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valency_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        ) = zip(*data_list)

        sequences = pad_sequence(sequences, batch_first=True, padding_value=get_id(PAD_TOKEN))
        count_sequences = pad_sequence(count_sequences, batch_first=True, padding_value=get_id(PAD_TOKEN))
        graph_mask_sequences = pad_sequence(graph_mask_sequences, batch_first=True, padding_value=get_id(PAD_TOKEN))
        valency_mask_sequences = pad_sequence(valency_mask_sequences, batch_first=True, padding_value=get_id(PAD_TOKEN))

        linear_loc_squares = pad_square(linear_loc_squares, padding_value=0)
        up_loc_squares = pad_square(up_loc_squares, padding_value=0)
        down_loc_squares = pad_square(down_loc_squares, padding_value=0)

        return (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valency_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        )
