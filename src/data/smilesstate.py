from rdkit import Chem
import networkx as nx
from tokenizers import Token

import torch
from torch.nn.utils.rnn import pad_sequence

from data.tokenize import get_tokentype, TokenType, get_bondorder
from copy import deepcopy


class SmilesState(object):
    def __init__(self, tokens):
        self.tokens = []
        self.tokentypes = []
        self.num_tokens = 0
        self.degrees = []
        self.numHs = []
        #
        self.max_atomidx = 0
        self.tokenidx2atomidx = dict()
        self.atomidx2tokenidxs = dict()

        #
        self.tokenidx2isopen = dict()
        self.open_branches = []
        self.open_ring_nums = dict()

        #
        self.segments = [[]]

        #
        self.atomidx2degree = dict()
        self.atomidx2numH = dict()
        

        #
        self.bosidx = None
        self.eosidx = None

        for token in tokens:
            self.update(token, debug=True)

    def update(self, token, debug=False):
        tokenidx = len(self.tokens)
        tokentype = get_tokentype(token)

        # Check grammar, put <eos> if fail
        incorrect = False
        if tokentype == TokenType.ATOM:
            if self.tokens[tokenidx - 1] != "<bos>" and self.tokentypes[tokenidx - 1] != TokenType.BOND:
                if debug:
                    print(self.tokens, token)
                    assert False

                incorrect = True
                
        elif tokentype == TokenType.BOND:
            if self.tokentypes[tokenidx - 1] not in [
                TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                ]:
                if debug:
                    print(self.tokens, token)
                    assert False

                incorrect = True
        
        elif tokentype == TokenType.BRANCH_START:
            if self.tokentypes[tokenidx - 1] not in [
                TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                ]:
                if debug:
                    print(self.tokens, token)
                    assert False

                incorrect = True

        elif tokentype == TokenType.BRANCH_END:
            if len(self.open_branches) == 0:
                if debug:
                    print(self.tokens, token)
                    assert False

                incorrect = True
        
        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                if self.tokentypes[tokenidx - 1] not in [
                    TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                    ]:
                    if debug:
                        print(self.tokens, token)
                        assert False

                    incorrect = True
            
            if token in self.open_ring_nums:
                if self.tokentypes[tokenidx - 1] != TokenType.BOND:
                    if debug:
                        print(self.tokens, token)
                        assert False

                    incorrect = True

        if incorrect:
            token = "<eos>"
            tokentype = TokenType.SPECIAL
        
        self.tokens.append(token)
        self.tokentypes.append(tokentype)
        
        # Mark bos and eos
        if token == "<bos>":
            self.bosidx = tokenidx if self.bosidx is None else self.bosidx

        if token == "<eos>":
            self.eosidx = tokenidx if self.eosidx is None else self.eosidx

        # Isopen logic
        self.tokenidx2isopen[tokenidx] = deepcopy(self.open_branches) + deepcopy(list(self.open_ring_nums.values()))

        # Segmentation logic
        if tokentype in [TokenType.ATOM, TokenType.BOND]:
            self.segments[-1].append(len(self.segments[-1]))

        elif tokentype in [TokenType.BRANCH_START, TokenType.BRANCH_END]:
            self.segments.append([0])
        
        elif tokentype == TokenType.RING_NUM:
            if token in self.open_ring_nums:
                self.segments[-1].append(len(self.segments[-1]))
            else:
                self.segments.append([0])

        elif tokentype == TokenType.SPECIAL:
            if token == "<bos>":
                self.segments[-1].append(len(self.segments[-1]))
                self.segments.append([])

            elif token == "<eos>":
                self.segments.append([0])

            else:
                self.segments[-1].append(len(self.segments[-1]))

        else:
            assert False

        # Equality logic
        def update_new_atom(tokenidx):
            self.max_atomidx += 1
            atomidx = self.max_atomidx
            self.tokenidx2atomidx[tokenidx] = atomidx
            self.atomidx2tokenidxs[self.max_atomidx] = [tokenidx]

        def update_seen_atom(prev_tokenidx, tokenidx):
            atomidx = self.tokenidx2atomidx[prev_tokenidx]
            self.tokenidx2atomidx[tokenidx] = atomidx
            self.atomidx2tokenidxs[atomidx].append(tokenidx)

        if tokentype in [TokenType.ATOM, TokenType.BOND, TokenType.SPECIAL]:
            update_new_atom(tokenidx)

        elif tokentype == TokenType.BRANCH_START:
            prev_tokenidx = tokenidx - 1
            self.open_branches.append(prev_tokenidx)
            update_seen_atom(prev_tokenidx, tokenidx)

        elif tokentype == TokenType.BRANCH_END:
            prev_tokenidx = self.open_branches.pop()
            update_seen_atom(prev_tokenidx, tokenidx)

        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                prev_tokenidx = tokenidx - 1
                self.open_ring_nums[token] = prev_tokenidx
                update_seen_atom(prev_tokenidx, tokenidx)

            else:
                prev_tokenidx = self.open_ring_nums.pop(token)
                update_seen_atom(prev_tokenidx, tokenidx)
        else:
            assert False

        # degree and numHs logic
        if tokentype == TokenType.ATOM:
            prev_tokenidx = tokenidx -1
            try:
                valence = Chem.MolFromSmiles(token).GetAtoms()[0].GetTotalValence()
            except:
                valence = Chem.MolFromSmiles(token.upper()).GetAtoms()[0].GetTotalValence()
                
            if self.tokens[prev_tokenidx] == "<bos>":
                degree = 0
                numH = valence
                
            elif self.tokentypes[prev_tokenidx] == TokenType.BOND:
                degree = 1
                bondorder = get_bondorder(self.tokens[prev_tokenidx])
                numH = valence - bondorder

            atomidx = self.tokenidx2atomidx[tokenidx]
            self.atomidx2degree[atomidx] = degree
            self.atomidx2numH[atomidx] = numH
        
        elif tokentype == TokenType.BOND:
            degree = -1
            numH = -1
            bond_order = get_bondorder(token)
            prev_tokenidx = tokenidx - 1
            
            if self.tokens[prev_tokenidx] in [
                TokenType.ATOM, TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM
                ]:
                atomidx = self.tokenidx2atomidx.get(prev_tokenidx, None)
                self.atomidx2degree[atomidx] += 1
                self.atomidx2numH[atomidx] -= bond_order
            
        elif tokentype in [TokenType.BRANCH_START, TokenType.BRANCH_END, TokenType.RING_NUM]:
            atomidx = self.tokenidx2atomidx.get(tokenidx, None)
            if atomidx is not None:
                degree = deepcopy(self.atomidx2degree[atomidx])
                numH = deepcopy(self.atomidx2numH[atomidx])
            else:
                degree = -1
                numH = -1
            

        elif tokentype == TokenType.SPECIAL:
            degree = -1
            numH = -1
            
        self.degrees.append(degree)
        self.numHs.append(numH)            

    def featurize(self, tokenizer):
        num_tokens = len(self.tokens)

        #
        token_sequence = torch.LongTensor([tokenizer.token_to_id(token) for token in self.tokens])

        #
        tokentype_sequence = torch.LongTensor(self.tokentypes)

        # 
        degree_sequence = torch.LongTensor([degree + 1 for degree in self.degrees])

        #
        numH_sequence = torch.LongTensor([numH * 2 + 1 for numH in self.numHs])
        numH_sequence[numH_sequence < 0] = 0

        #
        linedistance_square = (
            torch.abs(torch.arange(num_tokens).unsqueeze(1) - torch.arange(num_tokens).unsqueeze(0)) + 1
        )
        if self.bosidx is not None:
            linedistance_square[self.bosidx, :] = 0
            linedistance_square[:, self.bosidx] = 0

        elif self.eosidx is not None:
            linedistance_square[self.eosidx, :] = 0
            linedistance_square[:, self.eosidx] = 0

        #
        segments = [torch.LongTensor(segment) for segment in self.segments]
        segmentdistance_squares = [torch.abs(segment.unsqueeze(1) - segment.unsqueeze(0)) + 1 for segment in segments]
        distance_square = torch.block_diag(*segmentdistance_squares)

        #
        equality_square = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        for tokenidxs in self.atomidx2tokenidxs.values():
            mask = torch.zeros(num_tokens, dtype=torch.long)
            mask[tokenidxs] = 1
            equality_square += mask.unsqueeze(0) * mask.unsqueeze(1)
        
        equality_square[torch.arange(num_tokens).unsqueeze(1) == torch.arange(num_tokens).unsqueeze(0)] = 0

        #
        isopen_square = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        for tokenidx, isopen in self.tokenidx2isopen.items():
            isopen_square[tokenidx, isopen] = True

        ended = torch.tensor(True) if self.eosidx is not None else torch.tensor(False)

        return (
            token_sequence, 
            tokentype_sequence, 
            degree_sequence,
            numH_sequence,
            linedistance_square, 
            distance_square, 
            equality_square, 
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
            tokentype_sequences, 
            degree_sequences,
            numH_sequences,
            linedistance_squares, 
            distance_squares, 
            equality_squares, 
            isopen_squares, 
            endeds
        ) = zip(*data_list)

        token_sequences = pad_sequence(token_sequences, batch_first=True, padding_value=pad_id)
        tokentype_sequences = pad_sequence(tokentype_sequences, batch_first=True, padding_value=pad_id)
        degree_sequences= pad_sequence(degree_sequences, batch_first=True, padding_value=pad_id)
        numH_sequences= pad_sequence(numH_sequences, batch_first=True, padding_value=pad_id)
        
        linedistance_squares = pad_squares(linedistance_squares, pad_id)
        distance_squares = pad_squares(distance_squares, pad_id)
        equality_squares = pad_squares(equality_squares, pad_id)
        isopen_squares = pad_squares(isopen_squares, pad_id)
        
        endeds = torch.stack(endeds)

        return (
            token_sequences, 
            tokentype_sequences, 
            degree_sequences,
            numH_sequences,
            linedistance_squares, 
            distance_squares, 
            equality_squares, 
            isopen_squares, 
            endeds
        )


def pad_squares(squares, pad_id=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), pad_id, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
