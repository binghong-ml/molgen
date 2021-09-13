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


ATOM_OR_BOND_FEATURES = ["<pad>", "<bos>", "<eos>", "<atom>", "<bond>"]

ATOM_FEATURES = [
    "<pad>",
    (6, "CHI_TETRAHEDRAL_CCW", 0, 0),
    (6, "CHI_TETRAHEDRAL_CCW", 0, 1),
    (6, "CHI_TETRAHEDRAL_CW", 0, 0),
    (6, "CHI_TETRAHEDRAL_CW", 0, 1),
    (6, "CHI_UNSPECIFIED", -1, 1),
    (6, "CHI_UNSPECIFIED", -1, 2),
    (6, "CHI_UNSPECIFIED", 0, 0),
    (7, "CHI_UNSPECIFIED", -1, 0),
    (7, "CHI_UNSPECIFIED", -1, 1),
    (7, "CHI_UNSPECIFIED", 0, 0),
    (7, "CHI_UNSPECIFIED", 0, 1),
    (7, "CHI_UNSPECIFIED", 1, 0),
    (7, "CHI_UNSPECIFIED", 1, 1),
    (7, "CHI_UNSPECIFIED", 1, 2),
    (7, "CHI_UNSPECIFIED", 1, 3),
    (8, "CHI_UNSPECIFIED", -1, 0),
    (8, "CHI_UNSPECIFIED", 0, 0),
    (8, "CHI_UNSPECIFIED", 1, 0),
    (8, "CHI_UNSPECIFIED", 1, 1),
    (9, "CHI_UNSPECIFIED", 0, 0),
    (15, "CHI_TETRAHEDRAL_CCW", 0, 0),
    (15, "CHI_TETRAHEDRAL_CW", 0, 0),
    (15, "CHI_TETRAHEDRAL_CW", 0, 1),
    (15, "CHI_UNSPECIFIED", 0, 0),
    (15, "CHI_UNSPECIFIED", 0, 1),
    (15, "CHI_UNSPECIFIED", 0, 2),
    (15, "CHI_UNSPECIFIED", 1, 0),
    (15, "CHI_UNSPECIFIED", 1, 1),
    (16, "CHI_TETRAHEDRAL_CCW", 0, 0),
    (16, "CHI_TETRAHEDRAL_CW", 0, 0),
    (16, "CHI_TETRAHEDRAL_CW", 1, 0),
    (16, "CHI_UNSPECIFIED", -1, 0),
    (16, "CHI_UNSPECIFIED", 0, 0),
    (16, "CHI_UNSPECIFIED", 1, 0),
    (16, "CHI_UNSPECIFIED", 1, 1),
    (17, "CHI_UNSPECIFIED", 0, 0),
    (35, "CHI_UNSPECIFIED", 0, 0),
    (53, "CHI_UNSPECIFIED", 0, 0),
]

BOND_FEATURES = [
    "<pad>",
    ("AROMATIC", "ENDDOWNRIGHT"),
    ("AROMATIC", "ENDUPRIGHT"),
    ("AROMATIC", "NONE"),
    ("DOUBLE", "NONE"),
    ("SINGLE", "ENDDOWNRIGHT"),
    ("SINGLE", "ENDUPRIGHT"),
    ("SINGLE", "NONE"),
    ("TRIPLE", "NONE"),
]

QUEUE_FEATURES = ["<pad>"] + list(range(1, 50))

CHIRAL_TAG_DICT = {
    "CHI_UNSPECIFIED": Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    "CHI_TETRAHEDRAL_CW": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    "CHI_TETRAHEDRAL_CCW": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "CHI_OTHER": Chem.rdchem.ChiralType.CHI_OTHER,
}

BOND_TYPE_DICT = {
    "SINGLE": Chem.rdchem.BondType.SINGLE,
    "DOUBLE": Chem.rdchem.BondType.DOUBLE,
    "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    "AROMATIC": Chem.rdchem.BondType.AROMATIC,
}

BOND_DIR_DICT = {
    "NONE": Chem.rdchem.BondDir.NONE,
    "ENDUPRIGHT": Chem.rdchem.BondDir.ENDUPRIGHT,
    "ENDDOWNRIGHT": Chem.rdchem.BondDir.ENDDOWNRIGHT,
}

def get_atom_feature(atom):
    return atom.GetAtomicNum(), str(atom.GetChiralTag()), atom.GetFormalCharge(), atom.GetNumExplicitHs()

def get_bond_feature(bond):
    return str(bond.GetBondType()), str(bond.GetBondDir())

def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), feature=get_atom_feature(atom))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), feature=get_bond_feature(bond))

    return G

def nx_to_smiles(G):
    node_features = nx.get_node_attributes(G, "feature")
    edge_features = nx.get_edge_attributes(G, "feature")
    
    mol = Chem.RWMol()
    node_to_idx = dict()
    for node in G.nodes():
        atomic_num, chiral_tag, formal_charge, num_explicit_Hs = node_features[node]
        a=Chem.Atom(atomic_num)
        a.SetChiralTag(CHIRAL_TAG_DICT[chiral_tag])
        a.SetFormalCharge(formal_charge)
        a.SetNumExplicitHs(num_explicit_Hs)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for edge in G.edges():
        bond_type, bond_dir = edge_features[edge]
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]

        mol.AddBond(ifirst, isecond, BOND_TYPE_DICT[bond_type])
        mol.GetBondBetweenAtoms(ifirst, isecond).SetBondDir(BOND_DIR_DICT[bond_dir])

    smiles = Chem.MolToSmiles(mol)
    return smiles

def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares

class Data:
    def __init__(self, queue_scheme="bfs", max_queue_size=100):
        self.node_cnt = -1
        self.ended = False
        self.error = None
        
        #
        self.atom_or_bonds = []
        self.atomids = []
        self.bondids = []
        self.queueids = []
        
        #
        self.G = nx.Graph()

        #
        self.bos_node = None
        self.atom_to_node = dict()
        self.bond_to_node = dict()
        self.eos_node = None

        #
        self.node_to_depth = dict()
        self.atom_queues = []
        self.bond_queues = []
        
        #
        self.undirected_shortest_paths = []
        self.directed_shortest_paths = []
        
        #
        self.history = []
        
        #
        self.node_to_frontier = range(200)

        self.add_bos()
    
    def __len__(self):
        return len(self.G.nodes())

    def get_node_from_atom_queue(self, queueid):
        if len(self.atom_queues[-1]) < queueid:
            return None
        else:
            return self.atom_queues[-1][queueid - 1]
    
    def get_atom_queueid(self, node):
        if node in self.atom_queues[-1]:
            return self.atom_queues[-1].index(node) + 1
        else:
            return 0

    def update(self, atom_or_bond, atomid, bondid, queueid):
        self.history.append([atom_or_bond, atomid, bondid, queueid])

        if ATOM_OR_BOND_FEATURES[atom_or_bond] in ["<pad>", "<bos>"]:
            self.ended = True
            self.error = f"atom_or_bond: {ATOM_OR_BOND_FEATURES[atom_or_bond]} added"
        
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<eos>":
            self.add_eos()

        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<atom>":
            self.add_atom(atom_or_bond, atomid)
            
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<bond>":
            self.add_bond(atom_or_bond, bondid, queueid)

        else:
            self.ended = True
            self.error = f"atom_or_bond: {ATOM_OR_BOND_FEATURES[atom_or_bond]} unknown"

    def add_bos(self):
        if self.bos_node is not None:
            self.ended = True
            self.error = "<bos> already added."
            return
        
        self.atom_or_bonds.append(ATOM_OR_BOND_FEATURES.index("<bos>"))
        self.atomids.append(ATOM_FEATURES.index("<pad>"))
        self.bondids.append(BOND_FEATURES.index("<pad>"))
        self.queueids.append(QUEUE_FEATURES.index("<pad>"))

        self.node_cnt += 1
        self.bos_node = copy(self.node_cnt)
        self.G.add_node(self.bos_node)

        self.atom_queues.append([])
        self.bond_queues.append([])
        self.undirected_shortest_paths.append([])
        
    def add_eos(self):
        if self.eos_node is not None:
            self.ended = True
            self.error = "<eos> already added."
            return
        elif len(self.bond_queues[-1]) > 0:
            self.ended = True
            self.error = "<eos> before atom is closed."
            return
        else:
            self.ended = True

        #
        self.atom_or_bonds.append(ATOM_OR_BOND_FEATURES.index("<eos>"))
        self.atomids.append(ATOM_FEATURES.index("<pad>"))
        self.bondids.append(BOND_FEATURES.index("<pad>"))
        self.queueids.append(QUEUE_FEATURES.index("<pad>"))

        #
        self.node_cnt += 1
        self.eos_node = copy(self.node_cnt)
        self.G.add_node(self.eos_node)
        
        #
        self.atom_queues.append([])
        self.bond_queues.append([])
        self.undirected_shortest_paths.append([])

    def add_atom(self, atom_or_bond, atomid):
        if ATOM_FEATURES[atomid] in ["<pad>", "<bos>", "<eos>","<bond>"]:
            self.ended = True
            self.error = f"atomid: {ATOM_FEATURES[atomid]} added"
            return 

        #
        self.atom_or_bonds.append(atom_or_bond)
        self.atomids.append(atomid)
        self.bondids.append(BOND_FEATURES.index("<pad>"))
        self.queueids.append(QUEUE_FEATURES.index("<pad>"))

        #
        self.node_cnt += 1
        atom_node = copy(self.node_cnt)
        self.G.add_node(atom_node)
        
        #
        for bond_node in self.bond_queues[-1]:
            self.G.add_edge(bond_node, atom_node)

        #
        if len(self.bond_queues[-1]) > 0:
            depth = min([self.node_to_depth[bond_node] for bond_node in self.bond_queues[-1]])
        else:
            depth = 0

        self.node_to_depth[atom_node] = depth + 1

        #
        queue = [self.node_to_frontier[node] for node in self.atom_queues[-1] if self.node_to_depth[node] > depth - 1]
        queue.append(atom_node)
        self.atom_queues.append(queue)

        #
        self.bond_queues.append([])
        
        #
        paths = shortest_path(self.G, source=atom_node)
        self.undirected_shortest_paths.append([len(paths.get(node, [])) for node in range(self.node_cnt)])

    def add_bond(self, atom_or_bond, bondid, queueid):
        if BOND_FEATURES[bondid] in ["<pad>", "<bos>", "<eos>","<atom>"]:
            self.ended = True
            self.error = f"bondid: {bondid} added"
            return

        elif self.get_node_from_atom_queue(queueid) is None:
            self.ended = True
            self.error = "non-existing queueid"
            return
        
        elif queueid == 0:
            self.ended = True
            self.error = "zero queueid"
            return

        else:
            for bond_node in self.bond_queues[-1]:
                if self.get_node_from_atom_queue(queueid) in list(self.G[bond_node]):
                    self.ended = True
                    self.error = "duplicate node"
                    return

        #
        self.atom_or_bonds.append(atom_or_bond)
        self.atomids.append(ATOM_FEATURES.index("<pad>"))
        self.bondids.append(bondid)
        self.queueids.append(queueid)

        #
        self.node_cnt += 1
        bond_node = copy(self.node_cnt)
        self.G.add_node(bond_node)

        #
        atom_node = self.get_node_from_atom_queue(queueid)
        self.G.add_edge(atom_node, bond_node)
        
        #
        self.node_to_depth[bond_node] = self.node_to_depth[atom_node]

        #
        self.atom_queues.append([self.node_to_frontier[node] for node in self.atom_queues[-1]])

        #
        self.bond_queues.append([node for node in self.bond_queues[-1]] + [bond_node])

        #
        paths = shortest_path(self.G, source=atom_node)
        self.undirected_shortest_paths.append([len(paths.get(node, [])) for node in range(self.node_cnt)])

    def timeout(self):
        self.ended = True
        self.error = "timeout"

    def featurize(self):
        atom_or_bond_sequence = torch.LongTensor(self.atom_or_bonds)
        atomid_sequence = torch.LongTensor(self.atomids)
        bondid_sequence = torch.LongTensor(self.bondids)
        queueid_sequence = torch.LongTensor(self.queueids)

        adj_square = torch.LongTensor(nx.convert_matrix.to_numpy_array(self.G, weight="adj", dtype=np.int))
        
        def paths_to_square(path_lens, square_size):
            square = torch.zeros((square_size, square_size), dtype=torch.long)
            for idx in range(square_size):
                if len(path_lens[idx]) > 0:
                    square[idx, :len(path_lens[idx])] = torch.tensor(path_lens[idx]) + 1
            
            return square

        undirected_shortest_path_square = paths_to_square(self.undirected_shortest_paths, self.node_cnt + 1)
        
        def queues_to_square(queues, square_size):
            square = torch.zeros((square_size, square_size), dtype=torch.long)
            for idx, queue in enumerate(queues):
                square[idx, queue] = torch.arange(len(queue)) + 1
            
            return square

        atom_queueid_square = queues_to_square(self.atom_queues, self.node_cnt + 1)
        bond_queueid_square = queues_to_square(self.bond_queues, self.node_cnt + 1)

        return (
            atom_or_bond_sequence, 
            atomid_sequence,
            bondid_sequence,
            queueid_sequence, 
            undirected_shortest_path_square, 
            atom_queueid_square, 
            bond_queueid_square,
        )

    def to_smiles(self):
        if not self.ended:
            return None, "to_smiles for incomplete data"

        molG = nx.Graph()
    
        atom_cnt = 0
        node_to_atomidx = dict()
        node_attributes = dict()
        edge_attributes = dict()
        for node in range(self.node_cnt + 1):
            if ATOM_OR_BOND_FEATURES[self.atom_or_bonds[node]] == "<atom>":
                molG.add_node(atom_cnt)
                node_attributes[atom_cnt] = ATOM_FEATURES[self.atomids[node]]
                node_to_atomidx[node] = atom_cnt
                atom_cnt += 1
                
        for node in self.G:
            if ATOM_OR_BOND_FEATURES[self.atom_or_bonds[node]] == "<bond>":
                neighbors = [node2 for node2 in self.G.neighbors(node)]
                if len(list(neighbors)) < 2:
                    return None, "edge neighbor less than two"
                elif len(list(neighbors)) > 2:
                    return None, "edge neighbor more than two"

                atom_node0, atom_node1 = neighbors
                atomidx0 = node_to_atomidx[atom_node0]
                atomidx1 = node_to_atomidx[atom_node1]
                molG.add_edge(atomidx0, atomidx1)
                edge_attributes[atomidx0, atomidx1] = BOND_FEATURES[self.bondids[node]]

        nx.set_node_attributes(molG, node_attributes, 'feature')
        nx.set_edge_attributes(molG, edge_attributes, 'feature')
        
        return nx_to_smiles(molG), None
        
    @staticmethod
    def from_smiles(smiles):
        data = Data()
        molG = smiles_to_nx(smiles)
        node_features = nx.get_node_attributes(molG, "feature")
        edge_features = nx.get_edge_attributes(molG, "feature")
    
        def bfs_seq(G, start_id):
            dictionary = dict(nx.bfs_successors(G, start_id))
            start = [start_id]
            output = [start_id]
            while len(start) > 0:
                next = []
                while len(start) > 0:
                    current = start.pop(0)
                    neighbor = dictionary.get(current)
                    if neighbor is not None:
                        next = next + neighbor
                output = output + next
                start = next
            return output

        atom_to_node = dict()
        atom_seq = bfs_seq(molG, random.choice(list(molG.nodes())))
        for atom0 in atom_seq:
            for atom1 in molG[atom0]:
                if atom1 not in atom_to_node:
                    continue

                queueid = data.get_atom_queueid(atom_to_node[atom1])
                bond = (min(atom0, atom1), max(atom0, atom1))
                data.update(
                    atom_or_bond = ATOM_OR_BOND_FEATURES.index("<bond>"),
                    atomid = ATOM_FEATURES.index("<pad>"),
                    bondid = BOND_FEATURES.index(edge_features[bond]),
                    queueid=queueid
                )

                if data.error is not None:
                    print(data.error)
                    assert False

            data.update(
                atom_or_bond = ATOM_OR_BOND_FEATURES.index("<atom>"),
                atomid = ATOM_FEATURES.index(node_features[atom0]),
                bondid = BOND_FEATURES.index("<pad>"),
                queueid=0
            )
            if data.error is not None:
                print(data.error)
                assert False
            
            atom_to_node[atom0] = copy(data.node_cnt)
        
        data.add_eos()
        return data

    @staticmethod
    def collate(data_list):
        (
            atom_or_bond_sequences, 
            atomid_sequences, 
            bondid_sequences, 
            queueid_sequences, 
            adj_squares, 
            atom_queueid_squares, 
            bond_queueid_squares,
            ) = zip(*data_list)
    
        atom_or_bond_sequences = pad_sequence(atom_or_bond_sequences, batch_first=True, padding_value=0)
        atomid_sequences = pad_sequence(atomid_sequences, batch_first=True, padding_value=0)
        bondid_sequences = pad_sequence(bondid_sequences, batch_first=True, padding_value=0)
        queueid_sequences = pad_sequence(queueid_sequences, batch_first=True, padding_value=0)
        
        adj_squares = pad_square(adj_squares, padding_value=0)
        atom_queueid_squares = pad_square(atom_queueid_squares, padding_value=0)
        bond_queueid_squares = pad_square(bond_queueid_squares, padding_value=0)
        return (
            atom_or_bond_sequences, 
            atomid_sequences, 
            bondid_sequences, 
            queueid_sequences, 
            adj_squares, 
            atom_queueid_squares, 
            bond_queueid_squares,
            )


"""
def nx_to_sequence(G):
    node_features = nx.get_node_attributes(G, "feature")
    edge_features = nx.get_edge_attributes(G, "feature")
    
    sorted_nodes = sorted(G.nodes())
    atom_or_bond_sequence = [ATOM_OR_BOND_FEATURES.index("<bos>")]
    atomid_sequence = [ATOM_FEATURES.index("<bos>")]
    bondid_sequence = [ATOM_FEATURES.index("<bos>")]
    point_sequence = [-1]
    edge_end_sequence = [-1]
    
    node_to_idx = dict()
    for node in sorted_nodes:
        atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<atom>"))
        atomid_sequence.append(ATOM_FEATURES.index(node_features[node]))
        bondid_sequence.append(BOND_FEATURES.index("<misc>"))    
        edge_start_sequence.append(-1)
        edge_end_sequence.append(-1)

        node_to_idx[node] = len(atom_or_bond_sequence) - 1
        for edge_end in G[node]:
            if sorted_nodes.index(edge_end) < sorted_nodes.index(node):
                atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<bond>"))
                atomid_sequence.append(ATOM_FEATURES.index("<misc>"))
                bondid_sequence.append(BOND_FEATURES.index(edge_features[(edge_end, node)]))
                edge_start_sequence.append(node_to_idx[node])
                edge_end_sequence.append(node_to_idx[edge_end])

    atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<eos>"))
    atomid_sequence.append(ATOM_FEATURES.index("<eos>"))
    bondid_sequence.append(ATOM_FEATURES.index("<eos>"))
    edge_start_sequence.append(-1)
    edge_end_sequence.append(-1)

    atom_or_bond_sequence = torch.LongTensor(atom_or_bond_sequence)
    atomid_sequence = torch.LongTensor(atomid_sequence)
    bondid_sequence = torch.LongTensor(bondid_sequence)
    edge_start_sequence = torch.LongTensor(edge_start_sequence)
    edge_end_sequence = torch.LongTensor(edge_end_sequence)

    return atom_or_bond_sequence, atomid_sequence, bondid_sequence, edge_start_sequence, edge_end_sequence

def sequence_to_nx(
    atom_or_bond_sequence, atomid_sequence, bondid_sequence, edge_start_sequence, edge_end_sequence
    ):
    idx_to_atomidx = torch.cumsum(
        (atom_or_bond_sequence == ATOM_OR_BOND_FEATURES.index("<atom>")), dim=0
        )
    idx_to_atomidx = (idx_to_atomidx - 1).tolist()

    atom_or_bond_sequence = atom_or_bond_sequence.tolist()[1:-1]
    atomid_sequence = atomid_sequence.tolist()[1:-1]
    bondid_sequence = bondid_sequence.tolist()[1:-1]
    edge_start_sequence = edge_start_sequence.tolist()[1:-1]
    edge_end_sequence = edge_end_sequence.tolist()[1:-1]
    
    node = -1 
    G = nx.Graph()
    node_attributes = dict()
    edge_attributes = dict()
    for atom_or_bond, atomid, bondid, edge_start, edge_end in zip(
        atom_or_bond_sequence, atomid_sequence, bondid_sequence, edge_start_sequence, edge_end_sequence
        ):
        if ATOM_OR_BOND_FEATURES[atom_or_bond] == "<atom>":
            node += 1
            G.add_node(node)
            node_attributes[node] = ATOM_FEATURES[atomid]
        
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<bond>":
            G.add_edge(idx_to_atomidx[edge_end], idx_to_atomidx[edge_start])
            edge_attributes[idx_to_atomidx[edge_end], idx_to_atomidx[edge_start]] = BOND_FEATURES[bondid]
    

    nx.set_node_attributes(G, node_attributes, 'feature')
    nx.set_edge_attributes(G, edge_attributes, 'feature')

    return G
"""

"""
def nx_to_tsrs(G):
    nodes = sorted(G.nodes())
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    num_explicit_Hss = nx.get_node_attributes(G, 'num_explicit_Hs')
    
    atomic_nums_tsr = torch.LongTensor(np.array([atomic_nums[node] for node in nodes]))
    chiral_tags_tsr = torch.LongTensor(np.array([chiral_tags[node] for node in nodes]))
    formal_charges_tsr = torch.LongTensor(np.array([formal_charges[node] for node in nodes]))
    num_explicit_Hss_tsr = torch.LongTensor(np.array([num_explicit_Hss[node] for node in nodes]))

    node_tsrs = {
        'atomic_num': atomic_nums_tsr, 
        'chiral_tag': chiral_tags_tsr, 
        'formal_charge': formal_charges_tsr, 
        'num_explicit_Hs': num_explicit_Hss_tsr
        }

    #adj_np = nx.convert_matrix.to_numpy_array(G, dtype=np.int)
    #adj_tsr = torch.LongTensor(adj_np)
    shortestpath_len = torch.LongTensor(nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(G))
    bond_type_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_type', dtype=np.int))
    bond_dir_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_dir', dtype=np.int))
    bond_type_tsr[bond_type_tsr == 0] = FEATURES['bond_type'].index("<nobond>")
    bond_dir_tsr[bond_dir_tsr == 0] = FEATURES['bond_dir'].index("<nobond>")
    
    #bondidx_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bondidx', dtype=np.int))
    edge_tsrs = {
        #'adj': adj_tsr,
        'bond_type': bond_type_tsr, 
        'bond_dir': bond_dir_tsr,
        'shortest_path': shortestpath_len,
        #'bondidx': bondidx_tsr,
        }
    
    return node_tsrs, edge_tsrs

def tsrs_to_nx(node_tsrs, edge_tsrs):
    mask = node_tsrs['atomic_num'] != FEATURES['atomic_num'].index('<pad>')
    for key in node_tsrs:
        node_tsrs[key] = node_tsrs[key][mask]
    
    for key in edge_tsrs:
        edge_tsrs[key] = edge_tsrs[key][mask][:, mask]

    
    adj = (edge_tsrs['bond_type'] != FEATURES['bond_type'].index("<nobond>"))
    G = nx.from_numpy_array(adj.numpy())
    for node in G.nodes():
        for key, val in node_tsrs.items():
            G.nodes[node][key] = val[node]

    for edge in G.edges():
        for key, val in edge_tsrs.items():
            G.edges[edge][key] = val[edge]

    return G


def nx_to_smiles(G):
    mol = nx_to_mol(G)
    smiles = Chem.MolToSmiles(mol)
    return smiles

def smiles_to_tsrs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = mol_to_nx(mol)
    tsrs = nx_to_tsrs(G)
    return tsrs

def tsrs_to_smiles(node_tsrs, edge_tsrs):
    G = tsrs_to_nx(node_tsrs, edge_tsrs)
    try:
        mol = nx_to_mol(G)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None
    
"""
