# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import networkx as nx
from rdkit import Chem

ATOM_TOKENS = {
    "<C0>": (6, "CHI_TETRAHEDRAL_CCW", 0, 0),
    "<C1>": (6, "CHI_TETRAHEDRAL_CCW", 0, 1),
    "<C2>": (6, "CHI_TETRAHEDRAL_CW", 0, 0),
    "<C3>": (6, "CHI_TETRAHEDRAL_CW", 0, 1),
    "<C4>": (6, "CHI_UNSPECIFIED", -1, 1),
    "<C5>": (6, "CHI_UNSPECIFIED", -1, 2),
    "<C6>": (6, "CHI_UNSPECIFIED", 0, 0),
    "<N0>": (7, "CHI_UNSPECIFIED", -1, 0),
    "<N1>": (7, "CHI_UNSPECIFIED", -1, 1),
    "<N2>": (7, "CHI_UNSPECIFIED", 0, 0),
    "<N3>": (7, "CHI_UNSPECIFIED", 0, 1),
    "<N4>": (7, "CHI_UNSPECIFIED", 1, 0),
    "<N5>": (7, "CHI_UNSPECIFIED", 1, 1),
    "<N6>": (7, "CHI_UNSPECIFIED", 1, 2),
    "<N7>": (7, "CHI_UNSPECIFIED", 1, 3),
    "<O0>": (8, "CHI_UNSPECIFIED", -1, 0),
    "<O1>": (8, "CHI_UNSPECIFIED", 0, 0),
    "<O2>": (8, "CHI_UNSPECIFIED", 1, 0),
    "<O3>": (8, "CHI_UNSPECIFIED", 1, 1),
    "<F0>": (9, "CHI_UNSPECIFIED", 0, 0),
    "<P0>": (15, "CHI_TETRAHEDRAL_CCW", 0, 0),
    "<P1>": (15, "CHI_TETRAHEDRAL_CW", 0, 0),
    "<P2>": (15, "CHI_TETRAHEDRAL_CW", 0, 1),
    "<P3>": (15, "CHI_UNSPECIFIED", 0, 0),
    "<P4>": (15, "CHI_UNSPECIFIED", 0, 1),
    "<P5>": (15, "CHI_UNSPECIFIED", 0, 2),
    "<P6>": (15, "CHI_UNSPECIFIED", 1, 0),
    "<P7>": (15, "CHI_UNSPECIFIED", 1, 1),
    "<>": (16, "CHI_TETRAHEDRAL_CCW", 0, 0),
    "<>": (16, "CHI_TETRAHEDRAL_CW", 0, 0),
    "<>": (16, "CHI_TETRAHEDRAL_CW", 1, 0),
    "<>": (16, "CHI_UNSPECIFIED", -1, 0),
    "<>": (16, "CHI_UNSPECIFIED", 0, 0),
    "<>": (16, "CHI_UNSPECIFIED", 1, 0),
    "<>": (16, "CHI_UNSPECIFIED", 1, 1),
    "<>": (17, "CHI_UNSPECIFIED", 0, 0),
    "<>": (35, "CHI_UNSPECIFIED", 0, 0),
    "<>": (53, "CHI_UNSPECIFIED", 0, 0),
}
BOND_TOKENS = {
    "<:down>": ("AROMATIC", "ENDDOWNRIGHT"),
    "<:up>": ("AROMATIC", "ENDUPRIGHT"),
    "<:>": ("AROMATIC", "NONE"),
    "<=>": ("DOUBLE", "NONE"),
    "<-down>": ("SINGLE", "ENDDOWNRIGHT"),
    "<-up>": ("SINGLE", "ENDUPRIGHT"),
    "<->": ("SINGLE", "NONE"),
    "<#>": ("TRIPLE", "NONE"),
}

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

def get_atom_token(atom):
    feature = (
        atom.GetAtomicNum(), 
        str(atom.IsAromatic()), 
        str(atom.GetChiralTag()), 
        atom.GetFormalCharge(), 
        atom.GetNumExplicitHs()
    )
    return ATOMFEATURE2TOKEN[feature]

def get_bond_token(bond):
    feature = str(bond.GetBondType()), str(bond.GetBondDir())
    return BONDFEATURE2TOKEN[feature]

def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), token=get_atom_token(atom))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token=get_bond_token(bond))

    return G

def nx_to_smiles(G):
    node_tokens = nx.get_node_attributes(G, "token")
    edge_tokens = nx.get_edge_attributes(G, "token")
    
    mol = Chem.RWMol()
    node_to_idx = dict()
    for node in G.nodes():
        atomic_num, chiral_tag, formal_charge, num_explicit_Hs = node_tokens[node]
        a=Chem.Atom(atomic_num)
        a.SetChiralTag(CHIRAL_TAG_DICT[chiral_tag])
        a.SetFormalCharge(formal_charge)
        a.SetNumExplicitHs(num_explicit_Hs)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for edge in G.edges():
        bond_type, bond_dir = edge_tokens[edge]
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]

        mol.AddBond(ifirst, isecond, BOND_TYPE_DICT[bond_type])
        mol.GetBondBetweenAtoms(ifirst, isecond).SetBondDir(BOND_DIR_DICT[bond_dir])

    smiles = Chem.MolToSmiles(mol)
    return smiles