# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import numpy as np
from scipy import sparse
import networkx as nx
import torch

from rdkit import Chem
from copy import copy
# from rdkit import rdBase
# rdBase.DisableLog('rdApp.error')
from torch.nn.utils.rnn import pad_sequence


ATOM_OR_BOND_FEATURES = ["<pad>", "<bos>", "<eos>", "<atom>", "<bond>"]

ATOM_FEATURES = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<bond>",
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
    "<bos>",
    "<eos>",
    "<atom>",
    ("AROMATIC", "ENDDOWNRIGHT"),
    ("AROMATIC", "ENDUPRIGHT"),
    ("AROMATIC", "NONE"),
    ("DOUBLE", "NONE"),
    ("SINGLE", "ENDDOWNRIGHT"),
    ("SINGLE", "ENDUPRIGHT"),
    ("SINGLE", "NONE"),
    ("TRIPLE", "NONE"),
]

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
    def __init__(self):
        self.G = nx.Graph()
        self.node_cnt = -1
        self.ended = False
        self.error = None
        self.bond_frontiers = []
        self.point_frontiers = []
        self.history = []
        
        self.add_bos()
    
    def __len__(self):
        return len(self.G.nodes())
        
    def update(self, atom_or_bond, atom_id, bond_id, point_idx, debug=False):
        self.history.append([atom_or_bond, atom_id, bond_id, point_idx])

        if ATOM_OR_BOND_FEATURES[atom_or_bond] in ["<pad>", "<bos>"]:
            self.ended = True
            self.error = "atom_or_bond: <pad> added"
        
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<eos>":
            if len(self.bond_frontiers) > 0:
                self.ended = True
                self.error = "<eos> before atom is closed."
            else:
                self.add_eos()
                self.ended = True

        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<atom>":
            if ATOM_FEATURES[atom_id] in ["<pad>", "<bos>", "<eos>","<bond>"]:
                self.ended = True
                self.error = f"atom_id: {ATOM_FEATURES[atom_id]} added"
            else:
                self.add_atom(atom_or_bond, atom_id)
            
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<bond>":
            pointed_atom_or_bond = nx.get_node_attributes(self.G, "atom_or_bond")[point_idx]
            if ATOM_OR_BOND_FEATURES[pointed_atom_or_bond] != "<atom>":
                self.ended = True
                self.error = f"point_idx {point_idx}: pointing to non-atom {ATOM_OR_BOND_FEATURES[pointed_atom_or_bond]}"
                
            elif BOND_FEATURES[bond_id] in ["<pad>", "<bos>", "<eos>","<atom>"]:
                self.ended = True
                self.error = f"bond_id: {bond_id} added"

            elif point_idx in self.point_frontiers:
                self.ended = True
                self.error = f"duplicate point_idx"

            else:
                self.add_bond(atom_or_bond, bond_id, point_idx)

        else:
            self.ended = True
            self.error = f"atom_or_bond: {ATOM_OR_BOND_FEATURES[atom_or_bond]} unknown"

    def add_bos(self):
        self.node_cnt += 1
        self.G.add_node(
            copy(self.node_cnt), 
            atom_or_bond=ATOM_OR_BOND_FEATURES.index("<bos>"), 
            atom_id=ATOM_FEATURES.index("<bos>"), 
            bond_id=BOND_FEATURES.index("<bos>")
            )

    def add_eos(self):
        self.node_cnt += 1
        self.G.add_node(
            copy(self.node_cnt), 
            atom_or_bond=ATOM_OR_BOND_FEATURES.index("<eos>"), 
            atom_id=ATOM_FEATURES.index("<eos>"), 
            bond_id=BOND_FEATURES.index("<eos>")
            )

    def add_atom(self, atom_or_bond, atom_id):
        self.node_cnt += 1
        atom_node = copy(self.node_cnt)
        self.G.add_node(atom_node, atom_or_bond=atom_or_bond, atom_id=atom_id)
        for bond_node in self.bond_frontiers:
            self.G.add_edge(bond_node, atom_node, adj=1, frontier_adj=1)

        self.bond_frontiers = []
        self.point_frontiers = []

    def add_bond(self, atom_or_bond, bond_id, point_idx):
        self.node_cnt += 1
        bond_node = copy(self.node_cnt)
        self.G.add_node(bond_node, atom_or_bond=atom_or_bond, bond_id=bond_id, point_idx=point_idx)
        self.G.add_edge(point_idx, bond_node, adj=1, frontier_adj=1)
        
        for frontier_node in self.bond_frontiers:                
            self.G.add_edge(frontier_node, bond_node, adj=0, frontier_adj=1)
        
        self.bond_frontiers.append(bond_node)
        self.point_frontiers.append(point_idx)

    def timeout(self):
        self.ended = True
        self.error = "timeout"

    def featurize(self):
        atom_or_bonds = nx.get_node_attributes(self.G, "atom_or_bond")
        atom_ids = nx.get_node_attributes(self.G, "atom_id")
        bond_ids = nx.get_node_attributes(self.G, "bond_id")
        point_idxs = nx.get_node_attributes(self.G, "point_idx")
        
        atom_or_bond_sequence = []
        atom_id_sequence = []
        bond_id_sequence = []
        point_idx_sequence = []
        
        for node in range(self.node_cnt + 1):
            atom_or_bond = atom_or_bonds[node]
            atom_or_bond_sequence.append(atom_or_bond)

            atom_id = atom_ids.get(node, ATOM_FEATURES.index("<bond>"))
            atom_id_sequence.append(atom_id)

            bond_id = bond_ids.get(node, BOND_FEATURES.index("<atom>"))
            bond_id_sequence.append(bond_id)

            point_idx = point_idxs.get(node, -1)
            point_idx_sequence.append(point_idx)

        adj_square = nx.convert_matrix.to_numpy_array(self.G, weight="adj", dtype=np.int)
        frontier_adj_square = nx.convert_matrix.to_numpy_array(self.G, weight="frontier_adj", dtype=np.int)

        atom_or_bond_sequence = torch.LongTensor(atom_or_bond_sequence)
        atom_id_sequence = torch.LongTensor(atom_id_sequence)
        bond_id_sequence = torch.LongTensor(bond_id_sequence)
        point_idx_sequence = torch.LongTensor(point_idx_sequence)
        adj_square = torch.LongTensor(adj_square)
        frontier_adj_square = torch.LongTensor(frontier_adj_square)


        return (
            atom_or_bond_sequence, 
            atom_id_sequence,
            bond_id_sequence,
            point_idx_sequence, 
            adj_square, 
            frontier_adj_square
        )

    def to_smiles(self):
        if not self.ended:
            return None, "to_smiles for incomplete data"

        molG = nx.Graph()
        
        atom_or_bonds = nx.get_node_attributes(self.G, "atom_or_bond")
        atom_ids = nx.get_node_attributes(self.G, "atom_id")
        bond_ids = nx.get_node_attributes(self.G, "bond_id")
    
        atom_cnt = 0
        node_to_atom_idx = dict()
        node_attributes = dict()
        edge_attributes = dict()
        for node in self.G:
            if ATOM_OR_BOND_FEATURES[atom_or_bonds[node]] == "<atom>":
                molG.add_node(atom_cnt)
                node_attributes[atom_cnt] = ATOM_FEATURES[atom_ids[node]]
                node_to_atom_idx[node] = atom_cnt
                atom_cnt += 1
                
        for node in self.G:
            if ATOM_OR_BOND_FEATURES[atom_or_bonds[node]] == "<bond>":
                neighbors = [node2 for node2 in self.G.neighbors(node) if self.G.edges[(node, node2)]['adj'] > 0]
                if len(list(neighbors)) < 2:
                    return None, "edge neighbor less than two"
                elif len(list(neighbors)) > 2:
                    return None, "edge neighbor more than two"

                atom_node0, atom_node1 = neighbors
                atom_idx0 = node_to_atom_idx[atom_node0]
                atom_idx1 = node_to_atom_idx[atom_node1]
                molG.add_edge(atom_idx0, atom_idx1)
                edge_attributes[atom_idx0, atom_idx1] = BOND_FEATURES[bond_ids[node]]

        nx.set_node_attributes(molG, node_attributes, 'feature')
        nx.set_edge_attributes(molG, edge_attributes, 'feature')
        
        return nx_to_smiles(molG), None
        
    @staticmethod
    def from_smiles(smiles):
        data = Data()
        molG = smiles_to_nx(smiles)
        node_features = nx.get_node_attributes(molG, "feature")
        edge_features = nx.get_edge_attributes(molG, "feature")
    
        node_to_idx = dict()
        sorted_nodes = sorted(molG.nodes())
        for node in sorted_nodes:
            for edge_end in molG[node]:
                if sorted_nodes.index(edge_end) < sorted_nodes.index(node):
                    data.update(
                        atom_or_bond = ATOM_OR_BOND_FEATURES.index("<bond>"),
                        atom_id = ATOM_FEATURES.index("<bond>"),
                        bond_id = BOND_FEATURES.index(edge_features[(edge_end, node)]),
                        point_idx = node_to_idx[edge_end]
                    )
            
            data.update(
                atom_or_bond = ATOM_OR_BOND_FEATURES.index("<atom>"),
                atom_id = ATOM_FEATURES.index(node_features[node]),
                bond_id = BOND_FEATURES.index("<atom>"),
                point_idx = -1
            )
            node_to_idx[node] = len(data) - 1

        data.add_eos()
        return data

    @staticmethod
    def collate(data_list):
        (
            atom_or_bond_sequences, 
            atom_id_sequences, 
            bond_id_sequences, 
            point_idx_sequences, 
            adj_squares,
            frontier_adj_square
            ) = zip(*data_list)
    
        atom_or_bond_sequences = pad_sequence(atom_or_bond_sequences, batch_first=True, padding_value=0)
        atom_id_sequences = pad_sequence(atom_id_sequences, batch_first=True, padding_value=0)
        bond_id_sequences = pad_sequence(bond_id_sequences, batch_first=True, padding_value=0)
        point_idx_sequences = pad_sequence(point_idx_sequences, batch_first=True, padding_value=-1)
        
        adj_squares = pad_square(adj_squares, padding_value=0)
        frontier_adj_square = pad_square(frontier_adj_square, padding_value=0)
        return (
            atom_or_bond_sequences, 
            atom_id_sequences, 
            bond_id_sequences, 
            point_idx_sequences, 
            adj_squares, 
            frontier_adj_square
            )


"""
def nx_to_sequence(G):
    node_features = nx.get_node_attributes(G, "feature")
    edge_features = nx.get_edge_attributes(G, "feature")
    
    sorted_nodes = sorted(G.nodes())
    atom_or_bond_sequence = [ATOM_OR_BOND_FEATURES.index("<bos>")]
    atom_id_sequence = [ATOM_FEATURES.index("<bos>")]
    bond_id_sequence = [ATOM_FEATURES.index("<bos>")]
    point_sequence = [-1]
    edge_end_sequence = [-1]
    
    node_to_idx = dict()
    for node in sorted_nodes:
        atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<atom>"))
        atom_id_sequence.append(ATOM_FEATURES.index(node_features[node]))
        bond_id_sequence.append(BOND_FEATURES.index("<misc>"))    
        edge_start_sequence.append(-1)
        edge_end_sequence.append(-1)

        node_to_idx[node] = len(atom_or_bond_sequence) - 1
        for edge_end in G[node]:
            if sorted_nodes.index(edge_end) < sorted_nodes.index(node):
                atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<bond>"))
                atom_id_sequence.append(ATOM_FEATURES.index("<misc>"))
                bond_id_sequence.append(BOND_FEATURES.index(edge_features[(edge_end, node)]))
                edge_start_sequence.append(node_to_idx[node])
                edge_end_sequence.append(node_to_idx[edge_end])

    atom_or_bond_sequence.append(ATOM_OR_BOND_FEATURES.index("<eos>"))
    atom_id_sequence.append(ATOM_FEATURES.index("<eos>"))
    bond_id_sequence.append(ATOM_FEATURES.index("<eos>"))
    edge_start_sequence.append(-1)
    edge_end_sequence.append(-1)

    atom_or_bond_sequence = torch.LongTensor(atom_or_bond_sequence)
    atom_id_sequence = torch.LongTensor(atom_id_sequence)
    bond_id_sequence = torch.LongTensor(bond_id_sequence)
    edge_start_sequence = torch.LongTensor(edge_start_sequence)
    edge_end_sequence = torch.LongTensor(edge_end_sequence)

    return atom_or_bond_sequence, atom_id_sequence, bond_id_sequence, edge_start_sequence, edge_end_sequence

def sequence_to_nx(
    atom_or_bond_sequence, atom_id_sequence, bond_id_sequence, edge_start_sequence, edge_end_sequence
    ):
    idx_to_atom_idx = torch.cumsum(
        (atom_or_bond_sequence == ATOM_OR_BOND_FEATURES.index("<atom>")), dim=0
        )
    idx_to_atom_idx = (idx_to_atom_idx - 1).tolist()

    atom_or_bond_sequence = atom_or_bond_sequence.tolist()[1:-1]
    atom_id_sequence = atom_id_sequence.tolist()[1:-1]
    bond_id_sequence = bond_id_sequence.tolist()[1:-1]
    edge_start_sequence = edge_start_sequence.tolist()[1:-1]
    edge_end_sequence = edge_end_sequence.tolist()[1:-1]
    
    node = -1 
    G = nx.Graph()
    node_attributes = dict()
    edge_attributes = dict()
    for atom_or_bond, atom_id, bond_id, edge_start, edge_end in zip(
        atom_or_bond_sequence, atom_id_sequence, bond_id_sequence, edge_start_sequence, edge_end_sequence
        ):
        if ATOM_OR_BOND_FEATURES[atom_or_bond] == "<atom>":
            node += 1
            G.add_node(node)
            node_attributes[node] = ATOM_FEATURES[atom_id]
        
        elif ATOM_OR_BOND_FEATURES[atom_or_bond] == "<bond>":
            G.add_edge(idx_to_atom_idx[edge_end], idx_to_atom_idx[edge_start])
            edge_attributes[idx_to_atom_idx[edge_end], idx_to_atom_idx[edge_start]] = BOND_FEATURES[bond_id]
    

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
    
    #bond_idx_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_idx', dtype=np.int))
    edge_tsrs = {
        #'adj': adj_tsr,
        'bond_type': bond_type_tsr, 
        'bond_dir': bond_dir_tsr,
        'shortest_path': shortestpath_len,
        #'bond_idx': bond_idx_tsr,
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
