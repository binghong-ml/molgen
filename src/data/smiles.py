# https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

TOKEN2ATOMFEAT = {
    "<C0>": (6, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "<C1>": (6, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 1),
    "<C2>": (6, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "<C3>": (6, ChiralType.CHI_TETRAHEDRAL_CW, 0, 1),
    "<C4>": (6, ChiralType.CHI_UNSPECIFIED, -1, 1),
    "<C5>": (6, ChiralType.CHI_UNSPECIFIED, -1, 2),
    "<C6>": (6, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<N0>": (7, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "<N1>": (7, ChiralType.CHI_UNSPECIFIED, -1, 1),
    "<N2>": (7, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<N3>": (7, ChiralType.CHI_UNSPECIFIED, 0, 1),
    "<N4>": (7, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "<N5>": (7, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "<N6>": (7, ChiralType.CHI_UNSPECIFIED, 1, 2),
    "<N7>": (7, ChiralType.CHI_UNSPECIFIED, 1, 3),
    "<O0>": (8, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "<O1>": (8, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<O2>": (8, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "<O3>": (8, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "<F0>": (9, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<P0>": (15, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "<P1>": (15, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "<P2>": (15, ChiralType.CHI_TETRAHEDRAL_CW, 0, 1),
    "<P3>": (15, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<P4>": (15, ChiralType.CHI_UNSPECIFIED, 0, 1),
    "<P5>": (15, ChiralType.CHI_UNSPECIFIED, 0, 2),
    "<P6>": (15, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "<P7>": (15, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "<S0>": (16, ChiralType.CHI_TETRAHEDRAL_CCW, 0, 0),
    "<S1>": (16, ChiralType.CHI_TETRAHEDRAL_CW, 0, 0),
    "<S2>": (16, ChiralType.CHI_TETRAHEDRAL_CW, 1, 0),
    "<S3>": (16, ChiralType.CHI_UNSPECIFIED, -1, 0),
    "<S4>": (16, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<S5>": (16, ChiralType.CHI_UNSPECIFIED, 1, 0),
    "<S6>": (16, ChiralType.CHI_UNSPECIFIED, 1, 1),
    "<Cl0>": (17, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<Br0>": (35, ChiralType.CHI_UNSPECIFIED, 0, 0),
    "<I0>": (53, ChiralType.CHI_UNSPECIFIED, 0, 0),
}
ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}
TOKEN2BONDFEAT = {
    "<:down>": (BondType.AROMATIC, BondDir.ENDDOWNRIGHT),
    "<:up>": (BondType.AROMATIC, BondDir.ENDUPRIGHT),
    "<:>": (BondType.AROMATIC, BondDir.NONE),
    "<=>": (BondType.DOUBLE, BondDir.NONE),
    "<-down>": (BondType.SINGLE, BondDir.ENDDOWNRIGHT),
    "<-up>": (BondType.SINGLE, BondDir.ENDUPRIGHT),
    "<->": (BondType.SINGLE, BondDir.NONE),
    "<#>": (BondType.TRIPLE, BondDir.NONE),
}
BONDFEAT2TOKEN = {val: key for key, val in TOKEN2BONDFEAT.items()}
def get_atom_token(atom):
    feature = (
        atom.GetAtomicNum(),
        atom.GetChiralTag(), 
        atom.GetFormalCharge(), 
        atom.GetNumExplicitHs()
    )
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
        a=Chem.Atom(atomic_num)
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