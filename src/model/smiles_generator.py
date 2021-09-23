from joblib.parallel import delayed
from networkx.readwrite.gml import Token
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm
from joblib import Parallel, delayed

from data.smiles_dataset import TOKEN2ID, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TOKENS, MAX_LEN

from time import time

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class SmilesGenerator(nn.Module):
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, input_dropout, dropout,
    ):
        super(SmilesGenerator, self).__init__()
        self.nhead = nhead

        #
        self.token_embedding_layer = TokenEmbedding(len(TOKENS), emb_size)
        self.input_dropout = nn.Dropout(input_dropout)

        #
        self.distance_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(TOKENS))

    def forward(self, sequences):
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)

        #
        out = self.token_embedding_layer(sequences)
        out = self.input_dropout(out)

        #
        distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
        distance_squares[distance_squares > MAX_LEN] = MAX_LEN
        distance_squares = distance_squares.unsqueeze(0).repeat(batch_size, 1, 1)
        distance_squares = distance_squares.to(out.device)
        mask = self.distance_embedding_layer(distance_squares)
        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == TOKEN2ID[PAD_TOKEN]

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits = self.generator(out)

        return logits

    def decode(self, num_samples, max_len, device):
        sequences = torch.LongTensor([[TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break

            logits = self(sequences)
            preds = Categorical(logits=logits[:, -1]).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences
