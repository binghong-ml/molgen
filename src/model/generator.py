import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm

from data.util import ATOM_FEATURES, BOND_FEATURES, ATOM_OR_BOND_FEATURES, Data
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

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, sequence_length):
        return self.pos_embedding[:sequence_length, :]

class EdgeLogitLayer(nn.Module):
    def __init__(self, emb_size, hidden_dim, vocab_size):
        super(EdgeLogitLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, vocab_size * self.hidden_dim)
        self.linear1 = nn.Linear(emb_size, vocab_size * self.hidden_dim)

    def forward(self, x, atom_mask):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out0 = self.linear0(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)
        out1 = self.linear1(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)
        
        out0 = out0.permute(0, 2, 1, 3).reshape(-1, seq_len, self.hidden_dim)
        out1 = out1.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim, seq_len)
        logits = self.scale * torch.bmm(out0, out1).view(batch_size, self.vocab_size, seq_len, seq_len)
        
        #
        atom_mask = atom_mask.view(atom_mask.size(0), 1, 1, atom_mask.size(1)).repeat(1, self.vocab_size, seq_len, 1)
        logits = logits.masked_fill(~atom_mask, float("-inf"))

        #
        bool_mask = (torch.triu(torch.ones((seq_len, seq_len)), diagonal=0) == 0).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, seq_len, seq_len).repeat(batch_size, self.vocab_size, 1, 1).to(logits.device)
        logits = logits.masked_fill(bool_mask, float("-inf"))
        
        #
        logits = logits.permute(0, 2, 3, 1)  # batch_size x seq_len x seq_len x vocab_size 
        logits = logits.reshape(logits.size(0), logits.size(1), -1)
        
        return logits

class BaseGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        logit_hidden_dim,
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.position_embedding_layer = AbsolutePositionalEncoding(emb_size)
        self.atom_or_bond_embedding_layer = TokenEmbedding(len(ATOM_OR_BOND_FEATURES), emb_size)
        self.atom_id_embedding_layer = TokenEmbedding(len(ATOM_FEATURES), emb_size)
        self.bond_id_embedding_layer = TokenEmbedding(len(BOND_FEATURES), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.adj_embedding_layer = nn.Embedding(2, nhead)
        self.frontier_adj_embedding_layer = nn.Embedding(2, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.atom_or_bond_generator = nn.Linear(emb_size, len(ATOM_OR_BOND_FEATURES))
        self.atom_id_generator = nn.Linear(emb_size, len(ATOM_FEATURES))
        self.bond_id_and_end_generator = EdgeLogitLayer(emb_size, logit_hidden_dim, len(BOND_FEATURES))


    def forward(
        self, 
        atom_or_bond_sequences, 
        atom_id_sequences, 
        bond_id_sequences, 
        point_idx_sequences, 
        adj_squares,
        frontier_adj_squares, 
        ):
        batch_size = atom_or_bond_sequences.size(0)
        sequence_len = atom_or_bond_sequences.size(1)
        
        atom_or_bond_sequences = atom_or_bond_sequences.transpose(0, 1)
        atom_id_sequences = atom_id_sequences.transpose(0, 1)
        bond_id_sequences = bond_id_sequences.transpose(0, 1)
            
        #
        out = self.position_embedding_layer(sequence_len)
        out = out + self.atom_or_bond_embedding_layer(atom_or_bond_sequences)
        out = out + self.atom_id_embedding_layer(atom_id_sequences)
        out = out + self.bond_id_embedding_layer(bond_id_sequences)
        out = self.input_dropout(out)

        #
        mask = self.adj_embedding_layer(adj_squares)
        mask += self.frontier_adj_embedding_layer(frontier_adj_squares)
        mask = mask.permute(0, 3, 1, 2)
        
        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)
        
        #
        key_padding_mask = (atom_or_bond_sequences == ATOM_OR_BOND_FEATURES.index("<pad>")).transpose(0, 1)

        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        atom_or_bond_logits = self.atom_or_bond_generator(out)        
        atom_id_logits = self.atom_id_generator(out)
        atom_mask = (atom_or_bond_sequences == ATOM_OR_BOND_FEATURES.index("<atom>")).transpose(0, 1)
        bond_id_and_end_logits = self.bond_id_and_end_generator(out, atom_mask)
        
        return atom_or_bond_logits, atom_id_logits, bond_id_and_end_logits
    

    def decode(self, num_samples, max_len, device):
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []
        for idx in range(max_len):
            if len(data_list) == 0:
                break

            #
            (
                atom_or_bond_sequences, 
                atom_id_sequences, 
                bond_id_sequences, 
                point_idx_sequences, 
                adj_squares,
                frontier_adj_squares
            ) = Data.collate([data.featurize() for data in data_list])


            atom_or_bond_logits, atom_id_logits, bond_id_and_end_logits = self(
                atom_or_bond_sequences.to(device), 
                atom_id_sequences.to(device), 
                bond_id_sequences.to(device), 
                point_idx_sequences.to(device), 
                adj_squares.to(device),
                frontier_adj_squares.to(device),
            )

            atom_id_distribution = Categorical(logits=atom_id_logits[:, -1])
            next_atom_id = atom_id_distribution.sample()
            if idx > 0:
                atom_or_bond_distribution = Categorical(logits=atom_or_bond_logits[:, -1])
                next_atom_or_bond = atom_or_bond_distribution.sample()  
                bond_id_and_end_distribution = Categorical(logits=bond_id_and_end_logits[:, -1])
                next_bond_id_and_end = bond_id_and_end_distribution.sample()
                next_bond_id = next_bond_id_and_end % len(BOND_FEATURES)
                next_point_idx = next_bond_id_and_end // len(BOND_FEATURES)
                
                #for i in range(num_samples):
                #    print(next_point_idx.size())
                #    if next_point_idx[i] == 0:
                #        print()
                #        print(bond_id_and_end_logits.size())
                #        print(bond_id_and_end_logits[i, -1].reshape(-1, len(BOND_FEATURES)))
                #        assert False
            else:
                next_atom_or_bond = torch.full((num_samples, ), ATOM_OR_BOND_FEATURES.index("<atom>"), dtype=torch.long)
                next_bond_id = torch.full((num_samples, ), BOND_FEATURES.index("<atom>"), dtype=torch.long)
                next_point_idx = -torch.ones(num_samples, dtype=torch.long)
            
            for data, atom_or_bond, atom_id, bond_id, point_idx in zip(
                data_list, 
                next_atom_or_bond.tolist(), 
                next_atom_id.tolist(), 
                next_bond_id.tolist(), 
                next_point_idx.tolist()
                ):
                data.update(atom_or_bond, atom_id, bond_id, point_idx)
                        
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]
            
            if idx == max_len - 1:
                for data in data_list:
                    data.timeout()

        data_list = data_list + ended_data_list
        return data_list