import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm

from data.util import RING_TOKENS, VALUE_TOKENS, Data, get_value_id
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
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.position_embedding_layer = AbsolutePositionalEncoding(emb_size)
        self.val_embedding_layer = TokenEmbedding(len(VALUE_TOKENS), emb_size)
        self.ring_embedding_layer = TokenEmbedding(len(RING_TOKENS), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        
        #
        self.distance_embedding_layer = nn.Embedding(200, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(VALUE_TOKENS) * len(RING_TOKENS))
        

    def forward(self, val_sequences, ring_sequences, distance_squares):
        batch_size = val_sequences.size(0)
        sequence_len = val_sequences.size(1)
            
        #
        #out = self.position_embedding_layer(sequence_len)
        out = self.val_embedding_layer(val_sequences)
        out = out + self.ring_embedding_layer(ring_sequences)
        out = self.input_dropout(out)

        #
        mask = self.distance_embedding_layer(distance_squares)
        
        #
        #distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
        #distance_squares = distance_squares.view(1, sequence_len, sequence_len).repeat(batch_size, 1, 1)
        #mask += self.distance_embedding_layer(distance_squares.to(out.device))
        
        mask = mask.permute(0, 3, 1, 2)
        
        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)
        
        #
        key_padding_mask = (val_sequences == get_value_id("<pad>"))

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

            #
            val_sequences, ring_sequences, distance_squares = Data.collate([data.featurize() for data in data_list])


            logits = self(val_sequences.to(device), ring_sequences.to(device), distance_squares.to(device))

            next_pred = Categorical(logits=logits[:, -1]).sample()
            next_val_id = next_pred // len(RING_TOKENS)
            next_ring_id = next_pred % len(RING_TOKENS)

            for data, val_id, ring_id in zip(data_list, next_val_id.tolist(), next_ring_id.tolist()):
                try:
                    data.update(val_id, ring_id)
                except Exception as e:
                    data.error = e
                    data.ended = True
                        
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]
            
            if idx == max_len - 1:
                for data in data_list:
                    data.timeout()

        data_list = data_list + ended_data_list
        return data_list