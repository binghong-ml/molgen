from data.smilesstate import SmilesState
import torch
import torch.nn as nn
#from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen = 500):
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

class BaseGenerator(nn.Module):
    def __init__(
        self,
        tokenizer,
        num_encoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        use_linedistance,
        use_distance,
        use_equality,
        use_isopen, 
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead
        self.tokenizer = tokenizer
        vocab_size = self.tokenizer.get_vocab_size()
        
        #
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        
        #
        self.use_linedistance = use_linedistance
        if self.use_linedistance:
            self.linedistance_embedding_layer = nn.Embedding(500, nhead)
        
        #
        self.use_distance = use_distance
        if self.use_distance:
            self.distance_embedding_layer = nn.Embedding(500, nhead)
        
        #
        self.use_equality = use_equality
        if self.use_equality:
            self.equality_embedding_layer = nn.Embedding(2, nhead)
        
        #
        self.use_isopen = use_isopen
        if self.use_isopen:
            self.isopen_embedding_layer = nn.Embedding(2, nhead)

        #
        self.input_dropout = nn.Dropout(dropout)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, vocab_size)

        
    def forward(self, sequences, linedistances, distances, equalities, isopens):
        sequences = sequences.transpose(0, 1)
        sequence_len = sequences.size(0)
        
        #
        out = self.tok_emb(sequences)
        out = self.input_dropout(out)
        
        #
        batch_size = sequences.size(1)
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1)
        
        mask = bool_mask.float()
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.masked_fill(bool_mask == 1, float(0.0))
        mask = mask.to(out.device)
        
        if self.use_linedistance:
            linedistance_embedding = self.linedistance_embedding_layer(linedistances).permute(0, 3, 1, 2)
            mask[bool_mask == 1] = mask[bool_mask == 1] + linedistance_embedding[bool_mask == 1]
            
        if self.use_distance:
            distance_embedding = self.distance_embedding_layer(distances).permute(0, 3, 1, 2)
            mask[bool_mask == 1] = mask[bool_mask == 1] + distance_embedding[bool_mask == 1]
        
        if self.use_equality:
            equality_embedding = self.equality_embedding_layer(equalities).permute(0, 3, 1, 2)
            mask[bool_mask == 1] = mask[bool_mask == 1] + equality_embedding[bool_mask == 1]
        
        if self.use_isopen:
            isopen_embedding = self.equality_embedding_layer(isopens).permute(0, 3, 1, 2)
            mask[bool_mask == 1] = mask[bool_mask == 1] + isopen_embedding[bool_mask == 1]
        
        #
        mask = mask.view(-1, sequence_len, sequence_len)
        key_padding_mask = (sequences == self.tokenizer.token_to_id("<pad>")).transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        logits = self.generator(out)

        logits = logits.transpose(0, 1)
        
        return logits
    
    def decode(self, batch_size, max_len, device):
        states = [SmilesState(["<bos>"]) for _ in range(batch_size)]
        for _ in range(max_len):
            #
            tsrs = SmilesState.collate([state.featurize(self.tokenizer) for state in states])
            tsrs = [tsr.to(device) for tsr in tsrs]
            sequences, linedistances, distances, equalities, isopens, ended = tsrs

            #
            logits = self(sequences, linedistances, distances, isopens, equalities)[:, -1]
            distribution = torch.distributions.Categorical(logits=logits)
            next_ids = distribution.sample()
            next_ids[ended] = self.tokenizer.token_to_id("<pad>")
            
            #
            for next_id, state in zip(next_ids.tolist(), states):
                next_token = self.tokenizer.id_to_token(next_id)
                state.update(next_token)
        
        return states