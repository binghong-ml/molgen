# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

TOKEN2ATOMFEAT = {
    "[CH]": (6, 0, 1),
    "[CH2]": (6, 0, 2),
    "[CH-]": (6, -1, 1),
    "[CH2-]": (6, -1, 2),
    "[C]": (6, 0, 0),
    "[N-]": (7, -1, 0),
    "[NH-]": (7, -1, 1),
    "[N]": (7, 0, 0),
    "[NH]": (7, 0, 1),
    "[N+]": (7, 1, 0),
    "[NH+]": (7, 1, 1),
    "[NH2+]": (7, 1, 2),
    "[NH3+]": (7, 1, 3),
    "[O-]": (8, -1, 0),
    "[O]": (8, 0, 0),
    "[O+]": (8, 1, 0),
    "[OH+]": (8, 1, 1),
    "[F]": (9, 0, 0),
    "[P]": (15, 0, 0),
    "[PH]": (15, 0, 1),
    "[PH2]": (15, 0, 2),
    "[P+]": (15, 1, 0),
    "[PH+]": (15, 1, 1),
    "[S-]": (16, -1, 0),
    "[S]": (16, 0, 0),
    "[S+]": (16, 1, 0),
    "[SH]": (16, 0, 1),
    "[SH+]": (16, 1, 1),
    "[Cl]": (17, 0, 0),
    "[Br]": (35, 0, 0),
    "[I]": (53, 0, 0),
}
ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}
TOKEN2BONDFEAT = {
    "=": BondType.DOUBLE,
    "-": BondType.SINGLE,
    "#": BondType.TRIPLE,
}
BONDFEAT2TOKEN = {val: key for key, val in TOKEN2BONDFEAT.items()}

VALENCES = {
    "[C]": 4,
    "[CH]": 3,
    "[CH2]": 2,
    "[CH2-]": 1,
    "[CH-]": 2,
    "[N]": 3,
    "[N-]": 2,
    "[N+]": 4,
    "[NH]": 2,
    "[NH+]": 3,
    "[NH-]": 1,
    "[NH2+]": 2,
    "[NH3+]": 1,
    "[O]": 2,
    "[O-]": 1,
    "[O+]": 3,
    "[OH+]": 2,
    "[F]": 1,
    "[P]": 5,
    "[P+]": 4,
    "[PH]": 4,
    "[PH2]": 3,
    "[PH+]": 3,
    "[S]": 6,
    "[SH]": 3,
    "[SH+]": 2,
    "[S-]": 1,
    "[S+]": 3,
    "[Cl]": 1,
    "[Br]": 1,
    "[I]": 1,
}

BOND_ORDERS = {
    "-": 1,
    "=": 2,
    "#": 3,
}


def get_atom_token(atom):
    return ATOMFEAT2TOKEN[(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumExplicitHs())]


def get_bond_token(bond):
    return BONDFEAT2TOKEN[bond.GetBondType()]


def get_max_valence(atom_token):
    return VALENCES[atom_token]


def get_bond_order(bond_token):
    return BOND_ORDERS[bond_token]


def smiles2molgraph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), token=get_atom_token(atom))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token=get_bond_token(bond))

    return G


def molgraph2smiles(G):
    node_tokens = nx.get_node_attributes(G, "token")
    edge_tokens = nx.get_edge_attributes(G, "token")
    edge_tokens.update({(v, u): edge_tokens[u, v] for u, v in edge_tokens})

    mol = Chem.RWMol()
    node_to_idx = dict()
    for node in G.nodes():
        token = node_tokens[node]
        atomic_num, formal_charge, num_explicit_Hs = TOKEN2ATOMFEAT[node_tokens[node]]
        a = Chem.Atom(atomic_num)
        a.SetFormalCharge(formal_charge)
        a.SetNumExplicitHs(num_explicit_Hs)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for edge in G.edges():
        token = edge_tokens[edge]
        bond_type = TOKEN2BONDFEAT[token]
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]

        mol.AddBond(ifirst, isecond, bond_type)

    smiles = Chem.MolToSmiles(mol)
    
    return smiles
