# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

TOKEN2ATOMFEAT = {
    "C@@": (6, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "C@@H": (6, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 1),
    "C@": (6, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "C@H": (6, ChiralType.CHI_TETRAHEDRAL_CW, 0, 1),
    "CH2": (6, ChiralType.CHI_UNSPECIFIED, 0, 2),
    "CH-": (6, ChiralType.CHI_UNSPECIFIED, -1, 1),
    "CH2-": (6, ChiralType.CHI_UNSPECIFIED, -1, 2),
    "C": (6, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "N-": (7, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "NH-": (7, ChiralType.CHI_UNSPECIFIED, -1, 1),
    "N": (7, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "NH": (7, ChiralType.CHI_UNSPECIFIED, 0, 1),
    "N+": (7, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "NH+": (7, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "NH2+": (7, ChiralType.CHI_UNSPECIFIED, 1, 2),
    "NH3+": (7, ChiralType.CHI_UNSPECIFIED, 1, 3),
    "O-": (8, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "O": (8, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "O+": (8, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "OH+": (8, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "F": (9, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "P@@": (15, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "P@": (15, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "P@H": (15, ChiralType.CHI_TETRAHEDRAL_CW, 0, 1),
    "P": (15, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "PH": (15, ChiralType.CHI_UNSPECIFIED, 0, 1),
    "PH2": (15, ChiralType.CHI_UNSPECIFIED, 0, 2),
    "P+": (15, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "PH+": (15, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "S@@": (16, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "S@": (16, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "S@+": (16, ChiralType.CHI_TETRAHEDRAL_CW, 1, 0),
    "S-": (16, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "S": (16, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "S+": (16, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "SH": (16, ChiralType.CHI_UNSPECIFIED, 0, 1),
    "SH+": (16, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "Cl": (17, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "Br": (35, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "I": (53, ChiralType.CHI_UNSPECIFIED, 0, 0),
    
}
ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}
TOKEN2BONDFEAT = {
    ":\\": (BondType.AROMATIC, BondDir.ENDDOWNRIGHT),
    ":/": (BondType.AROMATIC, BondDir.ENDUPRIGHT),
    ":": (BondType.AROMATIC, BondDir.NONE),
    "=": (BondType.DOUBLE, BondDir.NONE),
    "-\\": (BondType.SINGLE, BondDir.ENDDOWNRIGHT),
    "-/": (BondType.SINGLE, BondDir.ENDUPRIGHT),
    "-": (BondType.SINGLE, BondDir.NONE),
    "#": (BondType.TRIPLE, BondDir.NONE),
}
BONDFEAT2TOKEN = {val: key for key, val in TOKEN2BONDFEAT.items()}


def get_atom_token(atom):
    feature = (atom.GetAtomicNum(), atom.GetChiralTag(), atom.GetFormalCharge(), atom.GetNumExplicitHs())
    return ATOMFEAT2TOKEN[feature]


def get_bond_token(bond):
    feature = bond.GetBondType(), bond.GetBondDir()
    return BONDFEAT2TOKEN[feature]


def smiles2molgraph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), token=get_atom_token(atom))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token=get_bond_token(bond))

    return G


def molgraph2smiles(G):
    node_tokens = nx.get_node_attributes(G, "token")
    edge_tokens = nx.get_edge_attributes(G, "token")

    mol = Chem.RWMol()
    node_to_idx = dict()
    for node in G.nodes():
        atomic_num, chiral_tag, formal_charge, num_explicit_Hs = TOKEN2ATOMFEAT[node_tokens[node]]
        a = Chem.Atom(atomic_num)
        a.SetChiralTag(chiral_tag)
        a.SetFormalCharge(formal_charge)
        a.SetNumExplicitHs(num_explicit_Hs)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for edge in G.edges():
        bond_type, bond_dir = TOKEN2BONDFEAT[edge_tokens[edge]]
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]

        mol.AddBond(ifirst, isecond, bond_type)
        mol.GetBondBetweenAtoms(ifirst, isecond).SetBondDir(bond_dir)

    smiles = Chem.MolToSmiles(mol)
    return smiles
