import torch
from torch.nn.utils.rnn import pad_sequence

from data.tokenize import get_tokentype, TokenType
from copy import deepcopy

class SmilesState(object):
    def __init__(self, tokens):
        self.tokens = []
        self.tokentypes = []
        self.num_tokens = 0
        self.max_setidx = 0
        self.tokenidx2setidx = dict()
        self.setidx2tokenidxs = dict()
        
        self.tokenidx2isopen = dict()
        self.open_branches = []
        self.open_ring_nums = dict()
        
        self.segments = [[]]
        
        self.bosidx = None
        self.eosidx = None

        for token in tokens:
            self.update(token)
        
    def update(self, token, verbose=0):
        tokenidx = len(self.tokens)
        tokentype = get_tokentype(token)

        self.tokens.append(token)
        self.tokentypes.append(tokentype)
        
        # Isopen logic
        self.tokenidx2isopen[tokenidx] = deepcopy(self.open_branches) + deepcopy(list(self.open_ring_nums.values()))
        
        # Equality logic
        if tokentype == TokenType.BRANCH_START:
            prev_tokenidx = tokenidx-1
            if prev_tokenidx < 0:
                if verbose > 0:
                    print("Invalid grammar: negative prev_tokenidx for BRANCH_START")
            
            elif self.tokentypes[prev_tokenidx] != TokenType.ATOM:
                if verbose > 0:
                    print("Invalid grammar: non-atom before BRANCH_START")
            
            else:
                self.open_branches.append(prev_tokenidx)
                self.update_equality(prev_tokenidx, tokenidx)

            
        elif tokentype == TokenType.BRANCH_END:
            if len(self.open_branches) == 0:
                if verbose > 0:
                    print("invalid grammar: BRANCH_END before BRANCH_START")
            else:
                prev_tokenidx = self.open_branches.pop()
                self.update_equality(prev_tokenidx, tokenidx)
            
        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                prev_tokenidx = tokenidx - 1
                if prev_tokenidx < 0:
                    if verbose > 0:
                        print("Invalid grammar: negative prev_tokenidx for RING_NUM")
                    
                elif self.tokentypes[prev_tokenidx] != TokenType.ATOM:
                    if verbose > 0:
                        print("Invalid grammar: non-atom before RING_NUM")
                    
                else:
                    self.open_ring_nums[token] = prev_tokenidx
                    self.update_equality(prev_tokenidx, tokenidx)
            else:
                prev_tokenidx = self.open_ring_nums.pop(token)
                self.update_equality(prev_tokenidx, tokenidx)
        
        elif token == "<bos>":
            self.bosidx = tokenidx if self.bosidx is None else self.bosidx
        
        elif token == "<eos>":
            self.eosidx = tokenidx if self.eosidx is None else self.eosidx            

        # Segmentation logic
        if tokentype == TokenType.BRANCH_START:
            if len(self.segments[-1]) == 0:
                if verbose > 0:
                    print("Invalid grammar: BRANCH_START at beginning of segment")
                
                self.segments[-1].append(len(self.segments[-1]))
            else:
                self.segments[-1].append(len(self.segments[-1]) - 1)
            
        elif tokentype == TokenType.BRANCH_END:
            self.segments.append([0])

        elif tokentype == TokenType.RING_NUM:
            if token not in self.open_ring_nums:
                if len(self.segments[-1]) == 0:
                    if verbose > 0:
                        print("Invalid grammar: RING_NUM at beginning of segment")
                    
                    self.segments[-1].append(len(self.segments[-1]))
                
                else:
                    self.segments[-1].append(len(self.segments[-1]) - 1)

            else:
                self.segments[-1].append(len(self.segments[-1]))
        
        elif token == "<bos>":
            self.segments[-1].append(len(self.segments[-1]))
            self.segments.append([])
        
        else:
            self.segments[-1].append(len(self.segments[-1]))
    
    def update_equality(self, prev_tokenidx, tokenidx):
        if prev_tokenidx not in self.tokenidx2setidx:
            self.max_setidx += 1
            self.tokenidx2setidx[prev_tokenidx] = self.max_setidx
            self.setidx2tokenidxs[self.max_setidx] = [prev_tokenidx]

        setidx = self.tokenidx2setidx[prev_tokenidx]
        self.tokenidx2setidx[tokenidx] = setidx
        self.setidx2tokenidxs[setidx].append(tokenidx)

    def featurize(self, tokenizer):
        num_tokens = len(self.tokens)
        
        #
        sequence = torch.LongTensor([tokenizer.token_to_id(token) for token in self.tokens])
        
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
        segmentdistance_squares = [
            torch.abs(segment.unsqueeze(1) - segment.unsqueeze(0)) + 1 for segment in segments
            ]
        distance_square = torch.block_diag(*segmentdistance_squares)

        #
        equality_square = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        for tokenidxs in self.setidx2tokenidxs.values():
            mask = torch.zeros(num_tokens, dtype=torch.long)
            mask[tokenidxs] = 1
            equality_square += mask.unsqueeze(0) * mask.unsqueeze(1)
        
        #
        isopen_square = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        for tokenidx, isopen in self.tokenidx2isopen.items():
            isopen_square[tokenidx, isopen] = True
        
        ended = torch.tensor(True) if self.eosidx is not None else torch.tensor(False)


        return sequence, linedistance_square, distance_square, equality_square, isopen_square, ended

    def get_smiles(self):
        bosidx = 0 if self.bosidx is None else self.bosidx
        eosidx = len(self.tokens) if self.eosidx is None else self.eosidx
        return "".join(self.tokens[bosidx+1:eosidx-1])

    @staticmethod
    def collate(data_list, pad_id=0):
        sequences, linedistance_squares, distance_squares, equality_squares, isopen_squares, ended_list = zip(*data_list)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        linedistance_squares = pad_squares(linedistance_squares, pad_id)
        distance_squares = pad_squares(distance_squares, pad_id)
        equality_squares = pad_squares(equality_squares, pad_id)
        isopen_squares = pad_squares(isopen_squares, pad_id)
        endeds = torch.stack(ended_list)

        return sequences, linedistance_squares, distance_squares, equality_squares, isopen_squares, endeds

def pad_squares(squares, pad_id=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), pad_id, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, :square.size(0), :square.size(1)] = square
    
    return batched_squares