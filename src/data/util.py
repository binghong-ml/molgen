# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import random
import numpy as np
from scipy import sparse
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
import torch

from rdkit import Chem
from copy import copy
# from rdkit import rdBase
# rdBase.DisableLog('rdApp.error')
from torch.nn.utils.rnn import pad_sequence

from data.smiles import smiles_to_nx, nx_to_smiles
from collections import defaultdict

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
BRANCH_TOKENS = ["<branch_start>", "<branch_next>", "<branch_end>"]
VALUE_TOKENS = SPECIAL_TOKENS + BRANCH_TOKENS + ATOM_TOKENS + BOND_TOKENS
RING_TOKENS = SPECIAL_TOKENS + ["<ring_start>"] + [("RING", ring_idx) for ring_idx in range(20)]

def get_value_id(token):
    return VALUE_TOKENS.index(token)

def get_value_token(id):
    return VALUE_TOKENS[id]

def get_ring_id(token):
    return RING_TOKENS.index(token)

def get_ring_token(id):
    return RING_TOKENS[id]

class Data:
    def __init__(self, queue_scheme="bfs", max_queue_size=100):
        self.G = nx.DiGraph()
        self._node_offset = 0
        self.pointer_node = -1
        
        self.ring_token2nodes = defaultdict(list)
        self.node2ring_token = dict()
        self.node2val_token = dict()

        self.branch_starts = []
        
        self.ended = False
        self.error = None
        
    def __len__(self):
        return len(self.G.nodes())

    def update(self, val_id, ring_id):
        val_token = get_value_token(val_id)
        ring_token = get_ring_token(ring_id)
        if val_token in ATOM_TOKENS:
            self.add_atom(val_token, ring_token)
        elif val_token in BOND_TOKENS:
            self.add_bond(val_token)
        elif val_token in BRANCH_TOKENS:
            self.add_branch(val_token)
        elif val_token == "<eos>":
            self.ended = True
        else:
            self.ended = True
            self.error = f"{val_token} token added"

    def add_atom(self, val_token, ring_token):
        new_node = self.create_node()
        
        self.G.add_node(new_node)
        if self.pointer_node > -1:
            self.G.add_edge(self.pointer_node, new_node)

        self.pointer_node = new_node
        
        self.node2val_token[new_node] = val_token
        self.node2ring_token[new_node] = ring_token
        if ring_token != get_ring_token(0):
            self.ring_token2nodes[ring_token].append(new_node)       
        
    def add_bond(self, val_token):
        if self.pointer_node == -1:
            self.ended = True
            self.error = "bond added when pointer is negative"
            return
        
        new_node = self.create_node()
        self.G.add_node(new_node)
        self.G.add_edge(self.pointer_node, new_node)
        self.pointer_node = new_node
        
        self.node2val_token[new_node] = val_token
    
    def add_branch(self, val_token):
        if self.pointer_node == -1:
            self.ended = True
            self.error = "branch added when pointer is negative"
            return

        new_node = self.create_node()
        
        self.G.add_node(new_node)
        self.G.add_edge(self.pointer_node, new_node)
        if val_token == "<branch_start>":
            self.pointer_node = new_node
            self.branch_starts.append(new_node)
        elif val_token == "<branch_next>":
            self.pointer_node = self.branch_starts[-1]
        elif val_token == "<branch_end>":
            self.branch_starts.pop()
            self.pointer_node = self.branch_starts[-1] if len(self.branch_starts) > 0 else -1

        self.node2val_token[new_node] = val_token

    def create_node(self):
        self._node_offset += 1
        return copy(self._node_offset)
        
    def to_smiles(self):
        mol_lineG = self.G.copy()
        for node in list(mol_lineG.nodes()):
            if self.node2val_token[node] in ["<branch_start>", "<branch_next>", "<branch_end>"]:
                predecessor_nodes = list(mol_lineG.predecessors(node))
                succesor_nodes = list(mol_lineG.successors(node))
                
                predecessor_node = predecessor_nodes[0]
                for succesor_node in succesor_nodes:
                    mol_lineG.remove_edge(node, succesor_node)
                    mol_lineG.add_edge(predecessor_node, succesor_node)
                
                mol_lineG.remove_node(node)
        
        
        mol_lineG = mol_lineG.to_undirected()
        for ring_nodes in self.ring_token2nodes.values():
            first_node, rest_nodes = ring_nodes[0], ring_nodes[1:]
            for rest_node in rest_nodes:
                neighbor_nodes = list(mol_lineG.neighbors(rest_node))
                mol_lineG.remove_edges_from([(node, rest_node) for node in neighbor_nodes])
                mol_lineG.remove_edges_from([(rest_node, node) for node in neighbor_nodes])
                mol_lineG.add_edges_from([(node, first_node) for node in neighbor_nodes])
                mol_lineG.remove_node(rest_node)
                        
        mol_G = nx.Graph()
        for node in mol_lineG:
            val_token = self.node2val_token[node]
            if val_token in ATOM_TOKENS:
                mol_G.add_node(node, token=val_token)
        
        for node in mol_lineG:
            val_token = self.node2val_token[node]
            if val_token in BOND_TOKENS:
                node0, node1 = list(mol_lineG.neighbors(node))
                mol_G.add_edge(node0, node1, token=val_token)

        smiles = nx_to_smiles(mol_G)
        return smiles

    @staticmethod
    def from_smiles(smiles):
        mol_G = smiles_to_nx(smiles)
        atom_tokens = nx.get_node_attributes(mol_G, "token")
        bond_tokens = nx.get_edge_attributes(mol_G, "token")
        #
        mol_lineG = nx.Graph()
        for node in mol_G.nodes():
            mol_lineG.add_node(node, token=atom_tokens[node])
        
        for edge in mol_G.edges():
            mol_lineG.add_node(edge, token=bond_tokens[edge])
            
            node0, node1 = edge
            mol_lineG.add_edge(node0, edge)
            mol_lineG.add_edge(node1, edge)
       
        spanning_tree = nx.minimum_spanning_tree(mol_lineG).edges()
        excluded_edges = [edge for edge in mol_lineG.edges() if edge not in spanning_tree]
        
        #
        node2ring, ring_cnt = dict(), 0
        for edge in excluded_edges:
            atom_node, bond_node = edge
            
            mol_lineG.remove_edge(atom_node, bond_node)
            new_atom_node = mol_lineG.number_of_nodes()
            mol_lineG.add_node(new_atom_node, token=atom_tokens[atom_node])
            mol_lineG.add_edge(new_atom_node, bond_node)

            if atom_node in node2ring:
                node2ring[new_atom_node] = node2ring[atom_node]
            else:
                ring_cnt += 1
                node2ring[atom_node] = ring_cnt
                node2ring[new_atom_node] = ring_cnt

        #        
        def keyfunc(idx):
            return (mol_G.degree(idx), mol_G.nodes[idx].get('token')[0] == 6, idx)

        tokens = nx.get_node_attributes(mol_lineG, "token")
        start = min(mol_G.nodes, key=keyfunc)
        dfs_successors = nx.dfs_successors(mol_lineG, source=start)

        def get_branch_start_id():
            return [get_value_id("<branch_start>"), 0]
        
        def get_branch_next_id():
            return [get_value_id("<branch_next>"), 0]
        
        def get_branch_end_id():
            return [get_value_id("<branch_end>"), 0]
        
        def get_sequence(node):
            val_id = get_value_id(tokens[node])
            ring_id = node2ring.get(node, 0) + 4

            seq = [[val_id, ring_id]]
            num_successors = len(dfs_successors.get(node, []))
            if num_successors == 1:
                next_node = dfs_successors[node][0]
                succ_seq = get_sequence(next_node)
                seq = seq + succ_seq
            
            elif num_successors > 1:
                seq.append(get_branch_start_id())
                for next_node in dfs_successors[node]:
                    seq += get_sequence(next_node)
                    seq.append(get_branch_next_id())
                
                seq.pop()
                seq.append(get_branch_end_id())
            
            return seq

        sequence = get_sequence(start)

        data = Data()
        for val_id, ring_id in sequence:
            data.update(val_id, ring_id)
        
        return data

    def featurize(self):
        nodes = [node for node in self.G]
        val_ids = (
            [get_value_id("<bos>")] 
            + [get_value_id(self.node2val_token[node]) for node in nodes] 
            + [get_value_id("<eos>")]
        )

        ring_ids = []
        for node in nodes:
            if node in self.node2ring_token:
                ring_ids.append(get_ring_id(self.node2ring_token[node]))
            else:
                ring_ids.append(get_ring_id(("RING", 0)))
            
        ring_ids = [get_ring_id("<bos>")] + ring_ids + [get_ring_id("<eos>")]

        distance_squares = torch.abs(
            torch.arange(len(nodes) + 2).unsqueeze(0) - torch.arange(len(nodes) + 2).unsqueeze(1)
            ) 
        distance_squares += 1
        distance_squares[0] = 0
        distance_squares[-1] = 0
        distance_squares[:, 0] = 0
        distance_squares[:, -1] = 0

        return torch.LongTensor(val_ids), torch.LongTensor(ring_ids), distance_squares

    @staticmethod
    def collate(data_list):
        val_sequences, ring_sequences, distance_squares = zip(*data_list)
        val_sequences = pad_sequence(val_sequences, batch_first=True, padding_value=get_value_id("<pad>"))
        ring_sequences = pad_sequence(ring_sequences, batch_first=True, padding_value=get_ring_id("<pad>"))
        distance_squares = pad_square(distance_squares, padding_value=0)

        return val_sequences, ring_sequences, distance_squares

def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
