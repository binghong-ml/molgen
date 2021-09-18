from joblib.parallel import delayed
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm
from joblib import Parallel, delayed

from data.util import PAD_TOKEN, TOKENS, RING_ID_START, RING_ID_END, Data, get_id, MAX_LEN
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class EdgeLogitLayer(nn.Module):
    def __init__(self, emb_size, hidden_dim):
        super(EdgeLogitLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, self.hidden_dim)
        self.linear1 = nn.Linear(emb_size, self.hidden_dim)

    def forward(self, x, sequences):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out0 = self.linear0(x).view(batch_size, seq_len, self.hidden_dim)
        
        out1_ = self.linear1(x).view(batch_size, seq_len, self.hidden_dim)
        
        index_ = sequences.masked_fill((sequences < RING_ID_START) | (sequences > RING_ID_END - 1), RING_ID_START-1)
        index_ = index_ - RING_ID_START + 1

        out1 = torch.zeros(batch_size, RING_ID_END - RING_ID_START + 1, self.hidden_dim).to(out1_.device)
        out1.scatter_(dim=1, index=index_.unsqueeze(-1).repeat(1, 1, self.hidden_dim), src=out1_)
        out1 = out1[:, 1:]
        out1 = out1.permute(0, 2, 1)
        logits = self.scale * torch.bmm(out0, out1)
        return logits

class BaseGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        disable_branchidx, 
        disable_loc,
        disable_edgelogit,
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.token_embedding_layer = TokenEmbedding(len(TOKENS), emb_size)
        self.disable_branchidx = disable_branchidx
        if not self.disable_branchidx:
            self.branch_embedding_layer = TokenEmbedding(MAX_LEN, emb_size)

        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)
        
        self.disable_loc = disable_loc
        if not self.disable_loc:
            self.up_loc_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)
            self.down_loc_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)
            self.branch_up_loc_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)
            self.branch_down_loc_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)
            self.branch_right_loc_embedding_layer = nn.Embedding(MAX_LEN+1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #

        #
        self.disable_edgelogit = disable_edgelogit
        if self.disable_edgelogit:
            self.generator = nn.Linear(emb_size, len(TOKENS))
        else:
            self.generator = nn.Linear(emb_size, len(TOKENS) - (RING_ID_END - RING_ID_START))
        self.ring_generator = EdgeLogitLayer(emb_size=emb_size, hidden_dim=emb_size)
        
        #
        

    def forward(self, batched_data):
        (
            sequences, 
            branch_sequences, 
            distance_squares, 
            up_loc_squares, 
            down_loc_squares, 
            branch_up_loc_squares, 
            branch_down_loc_squares, 
            branch_right_loc_squares, 
            pred_masks
        ) = batched_data
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
            
        #
        out = self.token_embedding_layer(sequences)
        if not self.disable_branchidx:
            out += self.branch_embedding_layer(branch_sequences)
            out /= 2

        out = self.input_dropout(out)

        #
        mask = self.distance_embedding_layer(distance_squares)
        cnt = 1
        if not self.disable_loc:
            mask += self.up_loc_embedding_layer(up_loc_squares)
            mask += self.down_loc_embedding_layer(down_loc_squares)
            mask += self.branch_up_loc_embedding_layer(branch_up_loc_squares)
            mask += self.branch_down_loc_embedding_layer(branch_down_loc_squares)
            mask += self.branch_right_loc_embedding_layer(branch_right_loc_squares)
            cnt += 5
        
        #if not self.disable_ring_loc:
        #    mask += self.ring_loc_embedding_layer(ring_pos_square)
        #    cnt += 1 

        mask /= cnt
        mask = mask.permute(0, 3, 1, 2)
        
        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)
        
        #
        key_padding_mask = (sequences == get_id(PAD_TOKEN))

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        if self.disable_edgelogit:
            logits = self.generator(out)        
        else:
            logits0 = self.generator(out)
            logits1 = self.ring_generator(out, sequences)
            logits = torch.cat([logits0, logits1], dim=2)
        
        logits = logits.masked_fill(pred_masks, float('-inf'))

        return logits
    

    def decode(self, num_samples, max_len, device):
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []

        parallel = Parallel(n_jobs=8)
        def _update_data(inp):
            data, id = inp
            data.update(id)
            return data
        
        for _ in tqdm(range(max_len)):
            if len(data_list) == 0:
                break

            batched_data = Data.collate([data.featurize() for data in data_list])
            batched_data = [tsr.to(device) for tsr in batched_data]
            logits = self(batched_data)
            preds = Categorical(logits=logits[:, -1]).sample()
            data_list = parallel(delayed(_update_data)(pair) for pair in zip(data_list, preds.tolist()))
                        
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]
            
        data_list = data_list + ended_data_list
        return data_list