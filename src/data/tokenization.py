import networkx as nx
from networkx.algorithms import euler
from networkx.algorithms.euler import eulerian_circuit
from rdkit import Chem
from itertools import product, combinations
from collections import defaultdict
import random

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


SPECIAL_TOKENS = [("SPECIAL", "<pad>"), ("SPECIAL", "<mask>"), ("SPECIAL", "<bos>"), ("SPECIAL", "<eos>")]
ATOM_TOKENS = [
    ("ATOM", 6, "CHI_TETRAHEDRAL_CCW", 0, 0),
    ("ATOM", 6, "CHI_TETRAHEDRAL_CCW", 0, 1),
    ("ATOM", 6, "CHI_TETRAHEDRAL_CW", 0, 0),
    ("ATOM", 6, "CHI_TETRAHEDRAL_CW", 0, 1),
    ("ATOM", 6, "CHI_UNSPECIFIED", -1, 1),
    ("ATOM", 6, "CHI_UNSPECIFIED", -1, 2),
    ("ATOM", 6, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 7, "CHI_UNSPECIFIED", -1, 0),
    ("ATOM", 7, "CHI_UNSPECIFIED", -1, 1),
    ("ATOM", 7, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 7, "CHI_UNSPECIFIED", 0, 1),
    ("ATOM", 7, "CHI_UNSPECIFIED", 1, 0),
    ("ATOM", 7, "CHI_UNSPECIFIED", 1, 1),
    ("ATOM", 7, "CHI_UNSPECIFIED", 1, 2),
    ("ATOM", 7, "CHI_UNSPECIFIED", 1, 3),
    ("ATOM", 8, "CHI_UNSPECIFIED", -1, 0),
    ("ATOM", 8, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 8, "CHI_UNSPECIFIED", 1, 0),
    ("ATOM", 8, "CHI_UNSPECIFIED", 1, 1),
    ("ATOM", 9, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 15, "CHI_TETRAHEDRAL_CCW", 0, 0),
    ("ATOM", 15, "CHI_TETRAHEDRAL_CW", 0, 0),
    ("ATOM", 15, "CHI_TETRAHEDRAL_CW", 0, 1),
    ("ATOM", 15, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 15, "CHI_UNSPECIFIED", 0, 1),
    ("ATOM", 15, "CHI_UNSPECIFIED", 0, 2),
    ("ATOM", 15, "CHI_UNSPECIFIED", 1, 0),
    ("ATOM", 15, "CHI_UNSPECIFIED", 1, 1),
    ("ATOM", 16, "CHI_TETRAHEDRAL_CCW", 0, 0),
    ("ATOM", 16, "CHI_TETRAHEDRAL_CW", 0, 0),
    ("ATOM", 16, "CHI_TETRAHEDRAL_CW", 1, 0),
    ("ATOM", 16, "CHI_UNSPECIFIED", -1, 0),
    ("ATOM", 16, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 16, "CHI_UNSPECIFIED", 1, 0),
    ("ATOM", 16, "CHI_UNSPECIFIED", 1, 1),
    ("ATOM", 17, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 35, "CHI_UNSPECIFIED", 0, 0),
    ("ATOM", 53, "CHI_UNSPECIFIED", 0, 0),
]
BOND_TOKENS = [
    ("BOND", "AROMATIC", "ENDDOWNRIGHT"),
    ("BOND", "AROMATIC", "ENDUPRIGHT"),
    ("BOND", "AROMATIC", "NONE"),
    ("BOND", "TRIPLE", "NONE"),
    ("BOND", "DOUBLE", "NONE"),
    ("BOND", "SINGLE", "ENDDOWNRIGHT"),
    ("BOND", "SINGLE", "ENDUPRIGHT"),
    ("BOND", "SINGLE", "NONE"),
    ("BOND", "<nobond>"),
]
VAL_TOKENS = SPECIAL_TOKENS + ATOM_TOKENS + BOND_TOKENS
ID2VAL_TOKEN = {id: token for id, token in enumerate(VAL_TOKENS)}
VAL_TOKEN2ID = {token: id for id, token in enumerate(VAL_TOKENS)}

POS_TOKENS = SPECIAL_TOKENS + [("ANCHOR", idx) for idx in range(-1, 50)]
ID2POS_TOKEN = {id: token for id, token in enumerate(POS_TOKENS)}
POS_TOKEN2ID = {token: id for id, token in enumerate(POS_TOKENS)}

def get_pad_token():
    return ("SPECIAL", "<pad>")

def get_bos_token():
    return ("SPECIAL", "<bos>")

def get_eos_token():
    return ("SPECIAL", "<eos>")

def get_nobond_token():
    return ("BOND", "<nobond>")

def get_atom_token(atom):
    return ("ATOM", atom.GetAtomicNum(), str(atom.GetChiralTag()), atom.GetFormalCharge(), atom.GetNumExplicitHs())

def get_bond_token(bond):
    return ("BOND", str(bond.GetBondType()), str(bond.GetBondDir()))

def get_pos_token(idx):
    return ("ANCHOR", idx)

def smiles2nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), token=get_atom_token(atom))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token=get_bond_token(bond))

    return G

def nx2sequence(G):
    nodes = list(G.nodes())
    eulerized_G = nx.eulerize(G.copy())
    source_node = random.choice(nodes)
    eulerian_circuit = list(nx.eulerian_circuit(eulerized_G, source=source_node))
    
    node_tokens = nx.get_node_attributes(G, "token")
    edge_tokens = nx.get_edge_attributes(G, "token")
    
    #
    anchor_idxs = list(range(50))
    #random.shuffle(anchor_idxs)
    anchored_nodes = [node for node in eulerized_G if eulerized_G.degree(node) > 2]
    if source_node not in anchored_nodes:
        anchored_nodes = anchored_nodes + [source_node]

    if len(anchored_nodes) > 50:
        print(len(anchored_nodes))
        assert False

    #
    val_token_seq = [get_bos_token()]
    pos_token_seq = [get_bos_token()]

    def add_node(node):
        val_token_seq.append(node_tokens[node])
        if node in anchored_nodes:
            anchor_idx = anchor_idxs[anchored_nodes.index(node)]
            pos_token_seq.append(get_pos_token(anchor_idx))
        else:
            pos_token_seq.append(get_pos_token(-1))
    
    def add_edge(edge):
        cur_node, next_node = edge
        maybe_G_edge = (min(cur_node, next_node), max(cur_node, next_node))
        val_token_seq.append(edge_tokens.get(maybe_G_edge, ("BOND", "<nobond>")))
        pos_token_seq.append(get_pos_token(-1))

    for edge in eulerian_circuit:
        cur_node, next_node = edge
        add_node(cur_node)
        add_edge(edge)
        
    add_node(next_node)
    
    val_token_seq.append(get_eos_token())
    pos_token_seq.append(get_eos_token())

    #print([token[1] for token in pos_token_seq[1:-1]])
    
    val_seq = [VAL_TOKEN2ID[token] for token in val_token_seq]
    pos_seq = [POS_TOKEN2ID[token] for token in pos_token_seq]

    return val_seq, pos_seq

def nx2smiles(G):
    if G is None:
        return None

    node_features = nx.get_node_attributes(G, "token")
    edge_features = nx.get_edge_attributes(G, "token")
    
    mol = Chem.RWMol()
    node_to_idx = dict()
    for node in G.nodes():
        _, atomic_num, chiral_tag, formal_charge, num_explicit_Hs = node_features[node]
        a=Chem.Atom(atomic_num)
        a.SetChiralTag(CHIRAL_TAG_DICT[chiral_tag])
        a.SetFormalCharge(formal_charge)
        a.SetNumExplicitHs(num_explicit_Hs)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    for edge in G.edges():
        _, bond_type, bond_dir = edge_features[edge]
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]

        mol.AddBond(ifirst, isecond, BOND_TYPE_DICT[bond_type])
        mol.GetBondBetweenAtoms(ifirst, isecond).SetBondDir(BOND_DIR_DICT[bond_dir])

    smiles = Chem.MolToSmiles(mol)

    return smiles

def sequence2nx(val_seq, pos_seq):
    val_token_seq = [ID2VAL_TOKEN[id] for id in val_seq]
    pos_token_seq = [ID2POS_TOKEN[id] for id in pos_seq]
    
    G = nx.Graph()
    node_cnt = 0
    idx2node = dict()
    pos_token2node = dict()
    for idx, (val_token, pos_token) in enumerate(zip(val_token_seq, pos_token_seq)):
        if pos_token == get_bos_token():
            pass

        if val_token in ATOM_TOKENS: 
            if pos_token == get_pos_token(-1):
                node_cnt += 1
                idx2node[idx] = node_cnt
                G.add_node(idx2node[idx], token=val_token)

            elif pos_token not in pos_token2node:
                node_cnt += 1
                idx2node[idx] = node_cnt
                pos_token2node[pos_token] = node_cnt
                G.add_node(idx2node[idx], token=val_token)

            elif pos_token in pos_token2node:
                idx2node[idx] = pos_token2node[pos_token]

                
    for idx, (val_token, pos_token) in enumerate(zip(val_token_seq, pos_token_seq)):
        if val_token in BOND_TOKENS and val_token != get_nobond_token():
            G.add_edge(idx2node[idx-1], idx2node[idx+1], token=val_token)

    return G
        



