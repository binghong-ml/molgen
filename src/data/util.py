# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

from itertools import combinations
import random
import numpy as np
from scipy import sparse
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path, shortest_path_length
import torch

from rdkit import Chem
from copy import copy
# from rdkit import rdBase
# rdBase.DisableLog('rdApp.error')
from torch.nn.utils.rnn import pad_sequence

from data.smiles import TOKEN2ATOMFEAT, TOKEN2BONDFEAT, molgraph2smiles, smiles2molgraph
from collections import defaultdict

BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
PAD_TOKEN = "[pad]"
SPECIAL_TOKENS = ["[pad]", "[mask]", "[bos]", "[eos]"]
ATOM_TOKENS = [token for token in TOKEN2ATOMFEAT]
BOND_TOKENS = [token for token in TOKEN2BONDFEAT]
BRANCH_START_TOKEN = "("
BRANCH_END_TOKEN = ")"
BRANCH_TOKENS = [BRANCH_START_TOKEN, BRANCH_END_TOKEN]
RING_TOKENS = [f"[R{idx}]" for idx in range(20)]
TOKENS = SPECIAL_TOKENS + ATOM_TOKENS + BOND_TOKENS + BRANCH_TOKENS + RING_TOKENS

def get_id(token):
    return TOKENS.index(token)

def get_token(id):
    return TOKENS[id]

def get_ring_token(idx):
    return f"[R{idx}]"

def get_ring_idx(token):
    return RING_TOKENS.index(token)

class Data:
    def __init__(self):
        self.ids = []
        self.tokens = []

        self.G = nx.DiGraph()
        self.position_G = nx.DiGraph()

        self._node_offset = -1
        self.branch_start_nodes = []
        self.ring_idx_to_nodes = defaultdict(list)
        
        self.pointer_node = None
        self.ended = False
        self.error = None
        
        self.bos_node = None
        self.eos_node = None

        self.branch_features = []
        self.update(get_id(BOS_TOKEN))

    def __len__(self):
        return len(self.G.nodes())

    def update(self, id):
        token = get_token(id)
        self.ids.append(id)
        self.tokens.append(token)
        new_node = self.create_node()
        self.G.add_node(new_node)
        self.position_G.add_node(new_node)
        
        if token in ATOM_TOKENS + BOND_TOKENS + RING_TOKENS:
            self.add_regular_token(new_node, token)
        elif token == BRANCH_START_TOKEN:
            self.add_branch_start_token(new_node)
        elif token == BRANCH_END_TOKEN:
            self.add_branch_end_token(new_node)
        elif token == BOS_TOKEN:
            self.add_bos_token(new_node)
        elif token == EOS_TOKEN:
            self.add_eos_token(new_node)
        else: 
            self.set_error(f"{token} token added.")

        self.save_current_features()

    def add_regular_token(self, new_node, token):
        if self.pointer_node is not None:
            if token in (ATOM_TOKENS + RING_TOKENS) and self.tokens[self.pointer_node] in (ATOM_TOKENS + RING_TOKENS):
                self.set_error("consecutive atom tokens")

            if token in BOND_TOKENS and self.tokens[self.pointer_node] in BOND_TOKENS:
                self.set_error("consecutive bond tokens")

            self.G.add_edge(self.pointer_node, new_node)

            #
            self.position_G.add_edge(self.pointer_node, new_node, weight=10**6)
            self.position_G.add_edge(new_node, self.pointer_node, weight=10**3)
        
        self.pointer_node = new_node

        if token in RING_TOKENS:
            ring_idx = get_ring_idx(token)
            if len(self.ring_idx_to_nodes[ring_idx]) == 2:
                self.set_error("more than 2 nodes with same ring_idx")

            self.ring_idx_to_nodes[ring_idx].append(new_node)

    def add_branch_start_token(self, new_node):
        if self.pointer_node is None:
            self.set_error("( added at start.")
            return 

        if self.tokens[self.pointer_node] not in ([BRANCH_START_TOKEN] + ATOM_TOKENS):
            self.set_error("( after non-{atom, )}")
            return
        
        #
        self.G.add_edge(self.pointer_node, new_node)

        #        
        self.position_G.add_edge(self.pointer_node, new_node, weight=10**6)
        self.position_G.add_edge(new_node, self.pointer_node, weight=10**3)

        prev_branch_start_nodes = [node for node in self.G.predecessors(self.pointer_node) if node != new_node]
        if len(prev_branch_start_nodes) > 0:
            prev_branch_start_node = max(prev_branch_start_nodes)
            self.position_G.add_edge(prev_branch_start_node, new_node, weight=1)
            
        self.branch_start_nodes.append(new_node)
        self.pointer_node = new_node
        
    def add_branch_end_token(self, new_node):
        if self.pointer_node is None:
            self.set_error(") added at start.")
            return

        if len(self.branch_start_nodes) == 0:
            self.set_error(") at depth 0.")
            return
            
        if self.tokens[self.pointer_node] not in (ATOM_TOKENS + RING_TOKENS):
            self.set_error(") after non-atom")
            return

        #
        self.G.add_edge(self.pointer_node, new_node)

        #
        self.position_G.add_edge(self.pointer_node, new_node, weight=10**6)
        self.position_G.add_edge(new_node, self.pointer_node, weight=10**3)

        branch_start_node = self.branch_start_nodes.pop()
        self.pointer_node = next(self.G.predecessors(branch_start_node))

    def add_bos_token(self, new_node):
        if self.pointer_node is not None:
            self.set_error("[bos] added at non-start.")
            return

        self.bos_node = new_node

    def add_eos_token(self, new_node):
        if self.pointer_node is None:
            self.set_error("[eos] added at start.")
            return 

        if len(self.branch_start_nodes) > 0:
            self.set_error("[eos] added before all branch closed.")
            return
        
        for ring_idx in self.ring_idx_to_nodes:
            if len(self.ring_idx_to_nodes[ring_idx]) < 2:
                self.set_error("[eos] added before all ring closed.")
                return

        if self.tokens[self.pointer_node] not in (ATOM_TOKENS + RING_TOKENS):
            self.set_error("ended with non-atom token")
            return
        
        self.ended = True
        self.pointer_node = None
        self.eos_node = new_node

    def save_current_features(self):
        num_nodes = self.G.number_of_nodes()
        branch_feature = torch.zeros(num_nodes, dtype=torch.long) 
        branch_feature[self.branch_start_nodes] = torch.arange(len(self.branch_start_nodes), 0, -1)
        self.branch_features.append(branch_feature)

    def set_error(self, msg):
        self.ended = True
        self.error = "".join(self.tokens) + " " + msg

    def create_node(self):
        self._node_offset += 1
        return copy(self._node_offset)
        
    def to_smiles(self):
        mollinegraph = self.G.copy()
        if self.bos_node is None:
            return "", "[bos] not added."

        mollinegraph.remove_node(self.bos_node)

        if self.eos_node is None:
            return "", "[eos] not added."

        mollinegraph.remove_node(self.eos_node)

        for node in list(mollinegraph.nodes()):
            if self.tokens[node] in [BRANCH_START_TOKEN, BRANCH_END_TOKEN]:
                predecessor_node = next(mollinegraph.predecessors(node))
                succesor_nodes = list(mollinegraph.successors(node))                
                for succesor_node in succesor_nodes:
                    mollinegraph.remove_edge(node, succesor_node)
                    mollinegraph.add_edge(predecessor_node, succesor_node)
                
                mollinegraph.remove_node(node)
        
        molgraph = nx.Graph()
        for node in mollinegraph:
            token = self.tokens[node]
            if token in ATOM_TOKENS + RING_TOKENS:
                molgraph.add_node(node, token=token)
        
        for node in mollinegraph:
            token = self.tokens[node]
            if token in BOND_TOKENS:
                node0, node1 = next(mollinegraph.successors(node)), next(mollinegraph.predecessors(node))
                molgraph.add_edge(node0, node1, token=token)

        
        for dangling_atom_node0, dangling_atom_node1 in self.ring_idx_to_nodes.values():
            dangling_bond_node0 = next(mollinegraph.predecessors(dangling_atom_node0))
            atom_node0 = next(mollinegraph.predecessors(dangling_bond_node0))

            dangling_bond_node1 = next(mollinegraph.predecessors(dangling_atom_node1))
            atom_node1 = next(mollinegraph.predecessors(dangling_bond_node1))
            
            molgraph.remove_nodes_from([dangling_atom_node0, dangling_atom_node1])
            molgraph.add_edge(atom_node0, atom_node1, token=self.tokens[dangling_bond_node0])

        smiles = molgraph2smiles(molgraph)

        return smiles, None

    @staticmethod
    def from_smiles(smiles):
        molgraph = smiles2molgraph(smiles)
        atom_tokens = nx.get_node_attributes(molgraph, "token")
        bond_tokens = nx.get_edge_attributes(molgraph, "token")
        bond_tokens.update({(node1, node0): val for (node0, node1), val in bond_tokens.items()})

        def keyfunc(idx):
            return (molgraph.degree(idx), molgraph.nodes[idx].get('token')[0] == 6, idx)

        tokens = nx.get_node_attributes(molgraph, "token")
        start = min(molgraph.nodes, key=keyfunc)
        
        dfs_successors = dict(nx.dfs_successors(molgraph, source=start))

        edges = set()
        for n_idx, n_jdxs in dfs_successors.items():
            for n_jdx in n_jdxs:
                edges.add((n_idx, n_jdx))
        
        ring_edges = [edge for edge in molgraph.edges if tuple(edge) not in edges]

        ring_nodes = list(range(molgraph.number_of_nodes(), molgraph.number_of_nodes() + len(ring_edges)))
        for idx, (ring_node, edge) in enumerate(zip(ring_nodes, ring_edges)):
            node0, node1 = edge

            atom_tokens[ring_node] = get_ring_token(idx)
            bond_tokens[node0, ring_node] = bond_tokens[node0, node1]
            bond_tokens[node1, ring_node] = bond_tokens[node0, node1]

            dfs_successors[node0] = [ring_node] + dfs_successors.get(node0, [])
            dfs_successors[node1] = [ring_node] + dfs_successors.get(node1, [])

        dfs_predecessors = dict()
        for node0 in dfs_successors:
            for node1 in dfs_successors[node0]:
                dfs_predecessors[node1] = node0

        tokens = []
        to_visit = [start]
        while to_visit:
            current = to_visit.pop()
            if current in [BRANCH_START_TOKEN, BRANCH_END_TOKEN]:
                tokens.append(current)
            else:
                if current in dfs_predecessors:
                    tokens.append(bond_tokens[dfs_predecessors[current], current])
            
                tokens.append(atom_tokens[current])
            
            next_nodes = dfs_successors.get(current, [])
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
                assert False
        
        data.update(get_id(EOS_TOKEN))
        return data

    def featurize(self):
        sequence = torch.LongTensor(self.ids)
        num_nodes = self.G.number_of_nodes()
        distance_square = torch.abs(torch.arange(num_nodes).unsqueeze(0) - torch.arange(num_nodes).unsqueeze(1)) + 1
        
        #
        shortest_path_lengths = dict(shortest_path_length(self.position_G, weight='weight'))
        up_loc_square = np.full((num_nodes, num_nodes), -1, dtype=int)
        down_loc_square = np.full((num_nodes, num_nodes), -1, dtype=int)
        right_loc_square = np.full((num_nodes, num_nodes), -1, dtype=int)

        for i in range(num_nodes):
            for j in range(num_nodes):
                path_len = shortest_path_lengths[i].get(j, None)
                if path_len is None:
                    continue
                
                up_loc_square[i, j], path_len = divmod(path_len, 10**6)
                down_loc_square[i, j], path_len = divmod(path_len, 10**3)
                right_loc_square[i, j] = path_len

        up_loc_square += 2
        down_loc_square += 2
        right_loc_square += 2

        up_loc_square = torch.LongTensor(up_loc_square)
        down_loc_square = torch.LongTensor(down_loc_square)
        right_loc_square = torch.LongTensor(right_loc_square)

        return sequence, distance_square, up_loc_square, down_loc_square, right_loc_square
        

    @staticmethod
    def collate(data_list):
        sequences, distance_squares, up_loc_squares, down_loc_squares, right_loc_squares = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=get_id(PAD_TOKEN))
        distance_squares = pad_square(distance_squares, padding_value=0)
        up_loc_squares = pad_square(up_loc_squares, padding_value=0)
        down_loc_squares = pad_square(down_loc_squares, padding_value=0)
        right_loc_squares = pad_square(right_loc_squares, padding_value=0)

        return sequences, distance_squares, up_loc_squares, down_loc_squares, right_loc_squares

def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
