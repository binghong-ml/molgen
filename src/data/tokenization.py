import networkx as nx
from networkx.algorithms import euler
from networkx.algorithms.euler import eulerian_circuit
from rdkit import Chem
from itertools import product, combinations
from collections import defaultdict
import random
from data.smiles import smiles2nx, nx2smiles

SPECIAL_TOKENS = ["<pad>", "<mask>", "<bos>", "<eos>"]
RING_TOKENS = [f"<ring_start_{idx}>" for idx in range(20)]
BRANCH_TOKENS = ["<branch_start>", "<branch_end>"]

def molgraph2moltree(molgraph):
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
        a.SetChiralTag(chiral_tag)
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
        



