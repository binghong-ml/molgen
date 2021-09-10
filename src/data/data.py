from rdkit import Chem
import networkx as nx
from tokenizers import Token

import torch
from torch.nn.utils.rnn import pad_sequence

from data.tokenize import get_tokentype, TokenType, get_bondorder
from copy import deepcopy

VALENCES = {"B": 3, "C": 4, "N": 5, "O": 2, "P": 5, "S": 6, "F": 1, "Cl": 1, "Br": 1, "I": 1}


class SourceData(object):
    def __init__(self, tokens):
        self.tokens = []
        self.tokentypes = []
        self.num_tokens = 0
        
        #
        self.graph = nx.Graph()
        
        #
        self.max_nodeidx = 0
        self.tokenidx2nodeidx = dict()
        self.nodeidx2tokenidxs = dict()

        #
        self.open_branches = []
        self.open_ring_nums = dict()
                
        #
        self.bosidx = None
        self.eosidx = None

        for token in tokens:
            self.update(token, debug=True)

    def update(self, token, debug=False):
        tokenidx = len(self.tokens)
        tokentype = get_tokentype(token)
        
        # Check grammar, put <eos> if fail
        if not self.is_correct_grammar(token, tokentype, tokenidx, debug):
            token = "<eos>"
            tokentype = TokenType.SPECIAL

        self.tokens.append(token)
        self.tokentypes.append(tokentype)
        
        # Mark bos and eos
        if token == "<bos>":
            self.bosidx = tokenidx if self.bosidx is None else self.bosidx

        if token == "<eos>":
            self.eosidx = tokenidx if self.eosidx is None else self.eosidx

        # Graph logic
        self.graph.add_node(tokenidx)
        if tokentype == TokenType.ATOM:
            #
            nodeidx = self.max_nodeidx
            self.tokenidx2nodeidx[tokenidx] = nodeidx
            self.nodeidx2tokenidxs[nodeidx] = [tokenidx]
            self.max_nodeidx += 1

            #
            anchor = tokenidx - 1
            
            #
            if self.tokens[anchor] != "<bos>":
                self.graph.add_edge(self.tokenidx2nodeidx[anchor], self.tokenidx2nodeidx[tokenidx], w=1)

            
        elif tokentype == TokenType.BOND:
            #
            nodeidx = self.max_nodeidx
            self.tokenidx2nodeidx[tokenidx] = nodeidx
            self.nodeidx2tokenidxs[nodeidx] = [tokenidx]
            self.max_nodeidx += 1

            #
            anchor = tokenidx - 1
            self.graph.add_edge(self.tokenidx2nodeidx[anchor], self.tokenidx2nodeidx[tokenidx], w=1)

        elif tokentype == TokenType.BRANCH_START:
            #
            anchor = tokenidx - 1
            nodeidx = self.tokenidx2nodeidx[tokenidx] = self.tokenidx2nodeidx[anchor]
            self.nodeidx2tokenidxs[nodeidx].append(tokenidx)
            self.open_branches.append(anchor)

        elif tokentype == TokenType.BRANCH_END:
            #
            anchor = self.open_branches.pop()
            nodeidx = self.tokenidx2nodeidx[tokenidx] = self.tokenidx2nodeidx[anchor]
            self.nodeidx2tokenidxs[nodeidx].append(tokenidx)
            
        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                #
                anchor = tokenidx - 1
                nodeidx = self.tokenidx2nodeidx[tokenidx] = self.tokenidx2nodeidx[anchor]
                self.nodeidx2tokenidxs[nodeidx].append(tokenidx)
                self.open_ring_nums[token] = tokenidx

            else:
                #
                anchor = self.open_ring_nums.pop(token)
                nodeidx = self.tokenidx2nodeidx[tokenidx] = self.tokenidx2nodeidx[anchor]
                self.nodeidx2tokenidxs[nodeidx].append(tokenidx)

                #
                self.graph.add_edge(anchor, tokenidx, w=1)
                
        elif tokentype == TokenType.SPECIAL:
            nodeidx = self.max_nodeidx
            self.tokenidx2nodeidx[tokenidx] = nodeidx
            self.nodeidx2tokenidxs[nodeidx] = [tokenidx]
            self.max_nodeidx += 1
            self.graph.add_node(nodeidx)
        
    def is_correct_grammar(self, token, tokentype, tokenidx, debug):
        if tokentype == TokenType.ATOM:
            if self.tokens[tokenidx - 1] != "<bos>" and self.tokentypes[tokenidx - 1] != TokenType.BOND:
                if debug:
                    print(self.tokens, token)
                    assert False

                print(self.tokens, token)
                return False
                
        elif tokentype == TokenType.BOND:
            if self.tokentypes[tokenidx - 1] not in [
                TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                ]:
                if debug:
                    print(self.tokens, token)
                    assert False

                print(self.tokens, token)
                return False
        
        elif tokentype == TokenType.BRANCH_START:
            if self.tokentypes[tokenidx - 1] not in [
                TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                ]:
                if debug:
                    print(self.tokens, token)
                    assert False

                print(self.tokens, token)
                return False

        elif tokentype == TokenType.BRANCH_END:
            if len(self.open_branches) == 0:
                if debug:
                    print(self.tokens, token)
                    assert False

                print(self.tokens, token)
                return False
        
        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                if self.tokentypes[tokenidx - 1] not in [
                    TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                    ]:
                    if debug:
                        print(self.tokens, token)
                        assert False

                    print(self.tokens, token)
                    return False
            
            if token in self.open_ring_nums:
                if self.tokentypes[tokenidx - 1] != TokenType.BOND:
                    if debug:
                        print(self.tokens, token)
                        assert False

                    print(self.tokens, token)
                    return False
        
        return True

    def featurize(self, tokenizer):
        #
        real_tokenidxs = [self.nodeidx2tokenidxs[nodeidx][0] for nodeidx in range(self.max_nodeidx)]  
        token_sequence = torch.LongTensor([tokenizer.token_to_id(self.tokens[tokenidx]) for tokenidx in real_tokenidxs])

        #
        path_length = list(nx.shortest_path_length(self.graph))
        path_length = [
            [path_length[nodeidx][1].get(nodeidx2, -1) for nodeidx2 in range(self.max_nodeidx)] 
            for nodeidx in range(self.max_nodeidx)
            ]
        distance_square = torch.LongTensor(path_length) + 1
        #distance_square[distance_square > 20] = 20
        
        return token_sequence, distance_square

    def get_smiles(self):
        bosidx = 0 if self.bosidx is None else self.bosidx
        eosidx = len(self.tokens) if self.eosidx is None else self.eosidx
        return "".join(self.tokens[bosidx + 1 : eosidx])

    @staticmethod
    def collate(data_list, pad_id=0):
        token_sequences, distance_squares = zip(*data_list)

        token_sequences = pad_sequence(token_sequences, batch_first=True, padding_value=pad_id)
        distance_squares = pad_squares(distance_squares, pad_id)
        
        return token_sequences, distance_squares

class TargetData(SourceData):
    def __init__(self, tokens):
        self.tokens = []
        self.tokentypes = []
        self.num_tokens = 0
        self.path_lens_list = []
        self.degrees_list = []
        self.bondorders_list = []
        
        #
        self.graph = nx.Graph()
        
        #
        self.max_atomidx = 0
        self.tokenidx2atomidx = dict()
        self.atomidx2tokenidxs = dict()

        #
        self.tokenidx2isopen = dict()
        self.open_branches = []
        self.open_ring_nums = dict()

        
        #
        self.atomidx2degree = dict()
        self.atomidx2bondorder = dict()
        
        #
        self.bosidx = None
        self.eosidx = None

        for token in tokens:
            self.update(token, debug=True)

    def update(self, token, debug=False):
        tokenidx = len(self.tokens)
        tokentype = get_tokentype(token)

        # Check grammar, put <eos> if fail
        if not self.is_correct_grammar(token, tokentype, tokenidx, debug):
            token = "<eos>"
            tokentype = TokenType.SPECIAL
        
        self.tokens.append(token)
        self.tokentypes.append(tokentype)
        
        # Mark bos and eos
        if token == "<bos>":
            self.bosidx = tokenidx if self.bosidx is None else self.bosidx

        if token == "<eos>":
            self.eosidx = tokenidx if self.eosidx is None else self.eosidx

        # Graph logic
        self.graph.add_node(tokenidx)
        if tokentype == TokenType.ATOM:
            #
            anchor = tokenidx - 1
            
            #
            if self.tokens[anchor] != "<bos>":
                self.graph.add_edge(anchor, tokenidx, w=1)

            #
            atomidx = self.max_atomidx
            self.tokenidx2atomidx[tokenidx] = atomidx
            self.atomidx2tokenidxs[atomidx] = [tokenidx]
            self.max_atomidx += 1
            
            #    
            if self.tokens[tokenidx -1] == "<bos>":
                degree = 0
                bondorder = 0
            
            elif self.tokentypes[tokenidx -1] == TokenType.BOND:
                degree = 1
                bondorder = get_bondorder(self.tokens[tokenidx -1])
                
            atomidx = self.tokenidx2atomidx[tokenidx]
            self.atomidx2degree[atomidx] = degree
            self.atomidx2bondorder[atomidx] = bondorder

        elif tokentype == TokenType.BOND:
            #
            anchor = tokenidx - 1
            
            #
            self.graph.add_edge(anchor, tokenidx, w=1)
            
            #

            #
            atomidx = self.tokenidx2atomidx[tokenidx - 1]
            self.atomidx2degree[atomidx] += 1
            self.atomidx2bondorder[atomidx] += get_bondorder(token)

        elif tokentype == TokenType.BRANCH_START:
            #
            anchor = tokenidx - 1

            #
            self.graph.add_edge(anchor, tokenidx, w=0)
            self.open_branches.append(tokenidx)

            #
            atomidx = self.tokenidx2atomidx[anchor]
            self.tokenidx2atomidx[tokenidx] = atomidx
            self.atomidx2tokenidxs[atomidx].append(tokenidx)

            #
            
        elif tokentype == TokenType.BRANCH_END:
            #
            anchor = self.open_branches.pop()

            #
            self.graph.add_edge(anchor, tokenidx, w=0)

            #
            atomidx = self.tokenidx2atomidx[anchor]
            self.tokenidx2atomidx[tokenidx] = atomidx
            self.atomidx2tokenidxs[atomidx].append(tokenidx)

            #

        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                #
                anchor = tokenidx - 1

                #
                self.graph.add_edge(anchor, tokenidx, w=0)
                self.open_ring_nums[token] = tokenidx

                #
                atomidx = self.tokenidx2atomidx[anchor]
                self.tokenidx2atomidx[tokenidx] = atomidx
                self.atomidx2tokenidxs[atomidx].append(tokenidx)
                
                #
            else:
                #
                anchor = self.open_ring_nums.pop(token)
                
                #
                self.graph.add_edge(anchor, tokenidx, w=1)
                
                #
                atomidx = self.tokenidx2atomidx[anchor]
                self.tokenidx2atomidx[tokenidx] = atomidx
                self.atomidx2tokenidxs[atomidx].append(tokenidx)
                
                #
        elif tokentype == TokenType.SPECIAL:
            pass

        else:
            assert False
        
        path_lens = nx.single_source_bellman_ford_path_length(self.graph, tokenidx, weight="w")    
        path_lens = [path_lens.get(i, -1) + 1 for i in range(len(self.tokens))]
        self.path_lens_list.append(torch.LongTensor(path_lens))
        
        # Isopen logic
        self.tokenidx2isopen[tokenidx] = deepcopy(self.open_branches) + deepcopy(list(self.open_ring_nums.values()))

        # 
        atomidxs = [self.tokenidx2atomidx.get(i, -1) for i in range(len(self.tokens))]
        self.degrees_list.append(torch.LongTensor([self.atomidx2degree.get(i, -1) + 1 for i in atomidxs]))
        self.bondorders_list.append(torch.LongTensor([self.atomidx2bondorder.get(i, -1) + 1 for i in atomidxs]))

    def featurize(self, tokenizer):
        num_tokens = len(self.tokens)

        #
        token_sequence = torch.LongTensor([tokenizer.token_to_id(token) for token in self.tokens])

        #
        distance_square = pad_sequence(self.path_lens_list, batch_first=True, padding_value=0)
        #distance_square[distance_square > 20] = 20

        #
        isopen_square = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        for tokenidx, isopen in self.tokenidx2isopen.items():
            isopen_square[tokenidx, isopen] = True
        
        #        
        ended = torch.tensor(True) if self.eosidx is not None else torch.tensor(False)

        return (
            token_sequence, 
            distance_square, 
            isopen_square, 
            ended
        )

    def get_smiles(self):
        bosidx = 0 if self.bosidx is None else self.bosidx
        eosidx = len(self.tokens) if self.eosidx is None else self.eosidx
        return "".join(self.tokens[bosidx + 1 : eosidx])

    @staticmethod
    def collate(data_list, pad_id=0):
        (
            token_sequences, 
            distance_squares, 
            isopen_squares, 
            endeds
        ) = zip(*data_list)

        token_sequences = pad_sequence(token_sequences, batch_first=True, padding_value=pad_id)
        distance_squares = pad_squares(distance_squares, pad_id)
        isopen_squares = pad_squares(isopen_squares, pad_id)        
        endeds = torch.stack(endeds)

        return token_sequences, distance_squares, isopen_squares, endeds

def pad_squares(squares, pad_id=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), pad_id, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
