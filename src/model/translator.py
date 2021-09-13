from networkx.readwrite.gml import Token
from data.data import TargetData
import torch
import torch.nn as nn

import math

from tqdm import tqdm
from data.data import TOKENS, TOKEN2ID, ID2TOKEN

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=500):
        super(AbsolutePositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, sequence_length):
        return self.pos_embedding[:sequence_length, :]


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class BaseEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
    ):
        super(BaseEncoder, self).__init__()
        self.nhead = nhead
        #
        self.tok_emb = TokenEmbedding(len(TOKENS), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(200, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src_batched_data):
        sequences, squares = src_batched_data
        sequence_len = sequences.size(1)
        
        #
        out = self.tok_emb(sequences)
        out = self.input_dropout(out)
        out = out.transpose(0, 1)
            
        #
        mask = self.distance_embedding_layer(squares)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, sequence_len, sequence_len)
        key_padding_mask = (sequences == TOKEN2ID["<pad>"])
        memory = self.transformer(out, mask, key_padding_mask)

        return memory, key_padding_mask

class BaseDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
    ):
        super(BaseDecoder, self).__init__()
        self.nhead = nhead

        #
        self.tok_emb = TokenEmbedding(len(TOKENS), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.distance_embedding_layer = nn.Embedding(200, nhead)

        #
        encoder_layer = nn.TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers, encoder_norm)

        self.generator = nn.Linear(emb_size, len(TOKENS))

    def forward(self, tgt_batched_data, memory, memory_key_padding_mask):
        sequences, distance_squares = tgt_batched_data        
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
            
        #
        out = self.tok_emb(sequences)
        out = self.input_dropout(out)

        #
        mask = self.distance_embedding_layer(distance_squares).permute(0, 3, 1, 2)
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)
        
        #
        key_padding_mask = (sequences == TOKEN2ID["<pad>"])

        out = out.transpose(0, 1)
        out = self.transformer(
            out, 
            memory, 
            tgt_mask=mask, 
            memory_mask=None,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
            )
        out = out.transpose(0, 1)
        logits = self.generator(out)
        
        return logits


class BaseTranslator(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout, 
    ):
        super(BaseTranslator, self).__init__()
        self.encoder = BaseEncoder(num_layers, emb_size, nhead, dim_feedforward, dropout)
        self.decoder = BaseDecoder(
            num_layers, 
            emb_size, 
            nhead,
            dim_feedforward, 
            dropout
        )

    def forward(self, src, tgt):
        memory, memory_key_padding_mask = self.encoder(src)
        logits = self.decoder(tgt, memory, memory_key_padding_mask)
        return logits

    def decode(self, src, max_len, device):
        batch_size = src[0].size(0)
        memory, memory_key_padding_mask = self.encoder(src)
        data_list = [TargetData() for _ in range(batch_size)]
        
        for _ in range(max_len):
            #
            tgt = TargetData.collate([data.featurize() for data in data_list])
            tgt = [tsr.to(device) for tsr in tgt]
            ended = torch.tensor([data.ended for data in data_list])
            #
            logits = self.decoder(tgt, memory, memory_key_padding_mask)[:, -1]
            distribution = torch.distributions.Categorical(logits=logits)
            next_ids = distribution.sample()
            next_ids[ended] = TOKEN2ID["<pad>"]

            #
            for next_id, data in zip(next_ids.tolist(), data_list):
                data.update(id=next_id)

        return data_list
