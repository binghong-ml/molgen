import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm

from data.util import PAD_TOKEN, TOKENS, Data, get_id
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=500):
        super(AbsolutePositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer("pos_embedding", pos_embedding.transpose(0, 1))

    def forward(self, sequence_length):
        return self.pos_embedding[:, :sequence_length]

class BaseGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        disable_loc,
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.position_embedding_layer = AbsolutePositionalEncoding(emb_size)
        self.token_embedding_layer = TokenEmbedding(len(TOKENS), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(200, nhead)
        
        self.disable_loc = disable_loc
        self.up_loc_embedding_layer = nn.Embedding(100, nhead)
        self.down_loc_embedding_layer = nn.Embedding(100, nhead)
        self.right_loc_embedding_layer = nn.Embedding(100, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(TOKENS))
        

    def forward(self, batched_data):
        sequences, distance_squares, up_loc_squares, down_loc_squares, right_loc_squares = batched_data
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
            
        #
        out = self.token_embedding_layer(sequences)
        out = self.input_dropout(out)

        #
        mask = self.distance_embedding_layer(distance_squares)
        if not self.disable_loc:
            mask += self.up_loc_embedding_layer(up_loc_squares)
            mask += self.down_loc_embedding_layer(down_loc_squares)
            mask += self.right_loc_embedding_layer(right_loc_squares)
        
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
        logits = self.generator(out)        
        return logits
    

    def decode(self, num_samples, max_len, device):
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []
        for idx in range(max_len):
            if len(data_list) == 0:
                break

            batched_data = Data.collate([data.featurize() for data in data_list])
            batched_data = [tsr.to(device) for tsr in batched_data]
            logits = self(batched_data)
            preds = Categorical(logits=logits[:, -1]).sample()
            for data, id in zip(data_list, preds.tolist()):
                data.update(id)
                        
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]
            
        data_list = data_list + ended_data_list
        return data_list