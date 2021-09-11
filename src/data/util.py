
#https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
#https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import numpy as np
from scipy import sparse
import networkx as nx
import torch

from rdkit import Chem

node_feature_names = ['atomic_num', 'chiral_tag', 'formal_charge', 'num_explicit_Hs']
edge_feature_names = ['bond_type', 'bond_dir']

allowable_features = {
    # Minimal atom features
    'atomic_num' : ['<pad>'] + list(range(1, 119)),
    'chiral_tag' : [
        '<pad>',
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    'formal_charge' : ['<pad>', -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_explicit_Hs' : ['<pad>', 0, 1, 2, 3, 4, 5, 6, 7, 8],
    # Minimal bond features
    'bond_type' : [
        '<pad>',
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'bond_dir': [
        '<pad>',
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        ],
}

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),       
            atomic_num=allowable_features['atomic_num'].index(atom.GetAtomicNum()),
            chiral_tag=allowable_features['chiral_tag'].index(atom.GetChiralTag()),
            formal_charge=allowable_features['formal_charge'].index(atom.GetFormalCharge()),
            num_explicit_Hs=allowable_features['num_explicit_Hs'].index(atom.GetNumExplicitHs())
            )
    
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=allowable_features['bond_type'].index(bond.GetBondType()),
            bond_dir=allowable_features['bond_dir'].index(bond.GetBondDir()),
            )
    
    return G


def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    num_explicit_Hss = nx.get_node_attributes(G, 'num_explicit_Hs')
    
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(allowable_features['atomic_num'][atomic_nums[node]])
        a.SetChiralTag(allowable_features['chiral_tag'][chiral_tags[node]])
        a.SetFormalCharge(allowable_features['formal_charge'][formal_charges[node]])
        a.SetNumExplicitHs(allowable_features['num_explicit_Hs'][num_explicit_Hss[node]])
        
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    bond_dirs = nx.get_edge_attributes(G, 'bond_dir')
    
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, allowable_features['bond_type'][bond_type])

        bond_dir = bond_dirs[first, second]
        new_bond = mol.GetBondBetweenAtoms(ifirst, isecond)
        new_bond.SetBondDir(allowable_features['bond_dir'][bond_dir])

    return mol

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

    adj_np = nx.convert_matrix.to_numpy_array(G, dtype=np.int)
    adj_tsr = torch.LongTensor(adj_np)
    shortestpath_len = torch.LongTensor(nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(G))
    bond_type_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_type', dtype=np.int))
    bond_dir_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_dir', dtype=np.int))
    #bond_idx_tsr = torch.LongTensor(nx.convert_matrix.to_numpy_array(G, weight='bond_idx', dtype=np.int))
    edge_tsrs = {
        'adj': adj_tsr,
        'shortestpath_len': shortestpath_len,
        'bond_type': bond_type_tsr, 
        'bond_dir': bond_dir_tsr,
        #'bond_idx': bond_idx_tsr,
        }
    
    return node_tsrs, edge_tsrs

def tsrs_to_nx(node_tsrs, edge_tsrs):
    G = nx.from_numpy_array(edge_tsrs['adj'].numpy())
    for node in G.nodes():
        for key, val in node_tsrs.items():
            G.nodes[node][key] = val[node]

    for edge in G.edges():
        for key, val in edge_tsrs.items():
            G.edges[edge][key] = val[edge]

    return G

def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_nx(mol)

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
    mol = nx_to_mol(G)
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None
        