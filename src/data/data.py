from rdkit import Chem
import networkx as nx
from tokenizers import Token

import torch
from torch.nn.utils.rnn import pad_sequence
from data.tokenize import tokenize, TokenType, get_tokentype

TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "#",
    "(",
    ")",
    "-",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    ":",
    "=",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "[C@@H]",
    "[C@@]",
    "[C@H]",
    "[C@]",
    "[CH-]",
    "[CH2-]",
    "[N+]",
    "[N-]",
    "[NH+]",
    "[NH-]",
    "[NH2+]",
    "[NH3+]",
    "[O+]",
    "[O-]",
    "[OH+]",
    "[P+]",
    "[P@@H]",
    "[P@@]",
    "[P@]",
    "[PH+]",
    "[PH2]",
    "[PH]",
    "[S+]",
    "[S-]",
    "[S@@+]",
    "[S@@]",
    "[S@]",
    "[SH+]",
    "[n+]",
    "[n-]",
    "[nH+]",
    "[nH]",
    "[o+]",
    "[s+]",
    "\\",
    "c",
    "n",
    "o",
    "s",
    # appear during randomization
    "[H]",
    "[S@+]",
]

ID2TOKEN = {id: token for id, token in enumerate(TOKENS)}
TOKEN2ID = {token: id for id, token in enumerate(TOKENS)}

PAD_ID = TOKEN2ID["<pad>"]
BOS_ID = TOKEN2ID["<bos>"]
EOS_ID = TOKEN2ID["<eos>"]

def pad_squares(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares

class SourceData(object):
    def __init__(self):
        #
        self.G = nx.Graph()
        self.ids = []
        
        #
        self.nodes_history = []
        self.branch_state = []
        self.ring_state = dict()
                
    def update(self, id=None, token=None):
        if id is None:
            id = TOKEN2ID[token]
        
        if token is None:
            token = ID2TOKEN[id]

        tokentype = get_tokentype(token)

        # Graph logic

        if tokentype in [TokenType.ATOM, TokenType.BOND]:
            # add node
            new_node = len(self.G.nodes())
            self.G.add_node(new_node)
            
            #
            if len(self.nodes_history) > 0:
                prev_node = self.nodes_history[-1]
                self.G.add_edge(prev_node, new_node)
            
            self.nodes_history.append(new_node)

            #
            self.ids.append(id)
            
        elif tokentype == TokenType.BRANCH_START:
            #
            prev_node = self.nodes_history[-1]
            self.branch_state.append(prev_node)
            self.nodes_history.append(prev_node)
            
        elif tokentype == TokenType.BRANCH_END:
            #
            anchor = self.branch_state.pop()
            self.nodes_history.append(anchor)

        elif tokentype == TokenType.RING_NUM:
            if token not in self.ring_state:
                prev_node = self.nodes_history[-1]
                self.ring_state[token] = prev_node
                self.nodes_history.append(prev_node)
            
            else:
                anchor = self.ring_state.pop(token)
                self.nodes_history.append(anchor)
                
        else: 
            assert False

    @staticmethod
    def from_smiles(smiles):
        tokens = tokenize(smiles)
        data = SourceData()
        for token in tokens:
            data.update(token=token)

        return data

    def featurize(self):
        sequence = torch.LongTensor(self.ids)
        path_length = torch.LongTensor(nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(self.G))
        distance_square = torch.LongTensor(path_length) + 1
        
        return sequence, distance_square

    @staticmethod
    def collate(data_list):
        sequences, squares = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        squares = pad_squares(squares, padding_value=0)
        return sequences, squares


class TargetData(object):
    def __init__(self):
        self.sequence = []
        self.ended = False
        self.update(token="<bos>")
        
    def get_branch_state(self):
        if len(self.branch_state_history) > 0:
            return self.branch_state_history[-1]
        else:
            return None

    def get_ring_state(self):
        if len(self.ring_state_history) > 0:
            return self.ring_state_history[-1]
        else:
            return None

    def update(self, id=None, token=None):
        if id is None:
            id = TOKEN2ID[token]
        
        token = ID2TOKEN[id]
        if token == "<pad>":
            pass
        elif token == "<eos>":
            self.ended = True
        else:
            self.sequence.append(id)
    
    @staticmethod
    def from_smiles(smiles, randomize=True):
        mol = Chem.MolFromSmiles(smiles)
        if randomize:
            smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False, allBondsExplicit=True)
        else:
            smiles = Chem.MolToSmiles(mol, canonical=True, allBondsExplicit=True)

        tokens = tokenize(smiles)
        
        data = TargetData()
        for token in tokens + ["<eos>"]:
            data.update(token=token)

        return data

    def to_smiles(self):
        smiles = "".join([ID2TOKEN[id] for id in self.sequence[1:-1]])
        return smiles

    def featurize(self):
        sequence = torch.LongTensor(self.sequence)
        arange_tsr = torch.arange(len(self.sequence))
        squares = torch.abs(arange_tsr.unsqueeze(0) - arange_tsr.unsqueeze(1))
        return sequence, squares
    
    @staticmethod
    def collate(data_list):
        sequences, squares = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=TOKEN2ID["<pad>"])
        squares = pad_squares(squares, padding_value=0)
        return sequences, squares
