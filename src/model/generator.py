import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm

from data.util import ATOM_FEATURES, BOND_FEATURES, ATOM_OR_BOND_FEATURES, QUEUE_FEATURES, Data
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

class EdgeLogitLayer(nn.Module):
    def __init__(self, emb_size, hidden_dim, vocab_size):
        super(EdgeLogitLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, vocab_size * self.hidden_dim)
        self.linear1 = nn.Linear(emb_size, vocab_size * self.hidden_dim)

    def forward(self, x, pos_square):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out0 = self.linear0(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)
        out1 = self.linear1(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)

        out0 = out0.permute(0, 2, 1, 3).reshape(-1, seq_len, self.hidden_dim)
        out1 = out1.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim, seq_len)
        logits_ = self.scale * torch.bmm(out0, out1).view(batch_size, self.vocab_size, seq_len, seq_len)
        logits_ = logits_.permute(0, 2, 3, 1) # batch_size x seq_len x seq_len x vocab_size
        
        #
        logits = torch.full((batch_size, seq_len, len(QUEUE_FEATURES), self.vocab_size), float('-inf')).to(out0.device)
        logits.scatter_(dim=2, index=pos_square.unsqueeze(-1).repeat(1, 1, 1, self.vocab_size), src=logits_)
        logits[:, :, 0] = float('-inf')
        logits = logits.reshape(batch_size, seq_len, -1)
        
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
        self.atomid_embedding_layer = TokenEmbedding(len(ATOM_FEATURES), emb_size)
        self.bondid_embedding_layer = TokenEmbedding(len(BOND_FEATURES), emb_size)
        
        #
        self.input_dropout = nn.Dropout(dropout)

        #
        self.adj_embedding_layer = nn.Embedding(200, nhead)
        self.atom_queueid_embedding_layer = nn.Embedding(len(QUEUE_FEATURES), nhead)
        self.bond_queueid_embedding_layer = nn.Embedding(len(QUEUE_FEATURES), nhead)
        
        #
        self.distance_embedding_layer = nn.Embedding(200, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.atom_or_bond_generator = nn.Linear(emb_size, len(ATOM_OR_BOND_FEATURES))
        self.atomid_generator = nn.Linear(emb_size, len(ATOM_FEATURES))
        self.bondid_and_queueid_generator = EdgeLogitLayer(emb_size, logit_hidden_dim, len(BOND_FEATURES))


    def forward(
        self,  
        atom_or_bond_sequences, 
        atomid_sequences,
        bondid_sequences,
        queueid_sequences, 
        adj_squares, 
        atom_queue_id_squares, 
        bond_queue_id_squares, 
        ):
        batch_size = atom_or_bond_sequences.size(0)
        sequence_len = atom_or_bond_sequences.size(1)
            
        #
        #out = self.position_embedding_layer(sequence_len)
        out = self.atom_or_bond_embedding_layer(atom_or_bond_sequences)
        out = out + self.atomid_embedding_layer(atomid_sequences)
        out = out + self.bondid_embedding_layer(bondid_sequences)
        out = self.input_dropout(out)

        #

        mask = self.adj_embedding_layer(adj_squares)
        mask += self.atom_queueid_embedding_layer(atom_queue_id_squares)
        mask += self.bond_queueid_embedding_layer(bond_queue_id_squares)
        
        #
        distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
        distance_squares = distance_squares.view(1, sequence_len, sequence_len).repeat(batch_size, 1, 1)
        mask += self.distance_embedding_layer(distance_squares.to(out.device))
        
        mask = mask.permute(0, 3, 1, 2)
        
        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)
        
        #
        key_padding_mask = (atom_or_bond_sequences == ATOM_OR_BOND_FEATURES.index("<pad>"))

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        atom_or_bond_logits = self.atom_or_bond_generator(out)        
        atomid_logits = self.atomid_generator(out)
        bondid_and_queueid_logits = self.bondid_and_queueid_generator(out, atom_queue_id_squares)
        
        
        #
        atom_or_bond_logits[:, :, ATOM_OR_BOND_FEATURES.index("<pad>")] = float("-inf")
        atom_or_bond_logits[:, :, ATOM_OR_BOND_FEATURES.index("<bos>")] = float("-inf")
        atom_or_bond_logits[:, 0, ATOM_OR_BOND_FEATURES.index("<bond>")] = float("-inf")
        
        return atom_or_bond_logits, atomid_logits, bondid_and_queueid_logits
    

    def decode(self, num_samples, max_len, device):
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []
        for idx in range(max_len):
            if len(data_list) == 0:
                break

            #
            (
                atom_or_bond_sequences, 
                atomid_sequences,
                bondid_sequences,
                queueid_sequences, 
                adj_squares, 
                atom_queue_id_squares, 
                bond_queue_id_squares,         
            ) = Data.collate([data.featurize() for data in data_list])


            atom_or_bond_logits, atomid_logits, bondid_and_queueid_logits = self(
                atom_or_bond_sequences.to(device), 
                atomid_sequences.to(device),
                bondid_sequences.to(device),
                queueid_sequences.to(device), 
                adj_squares.to(device), 
                atom_queue_id_squares.to(device), 
                bond_queue_id_squares.to(device), 
            )

            def sample_from_logits(logits, mask=None):
                if mask is not None:
                    sample = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                    sample[mask] = Categorical(logits=logits[mask, -1]).sample()
                    return sample
                else:
                    return Categorical(logits=logits[:, -1]).sample()

            next_atom_or_bond = sample_from_logits(atom_or_bond_logits) 
            next_atomid = sample_from_logits(atomid_logits) 
            next_bondid_and_queueid = sample_from_logits(
                bondid_and_queueid_logits, next_atom_or_bond==ATOM_OR_BOND_FEATURES.index("<bond>")
                )
            next_bondid = next_bondid_and_queueid % len(BOND_FEATURES)
            next_queueid = next_bondid_and_queueid // len(BOND_FEATURES)

            for data, atom_or_bond, atomid, bondid, queueid in zip(
                data_list, 
                next_atom_or_bond.tolist(), 
                next_atomid.tolist(), 
                next_bondid.tolist(), 
                next_queueid.tolist()
                ):
                data.update(atom_or_bond, atomid, bondid, queueid)
                        
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]
            
            if idx == max_len - 1:
                for data in data_list:
                    data.timeout()

        data_list = data_list + ended_data_list
        return data_list