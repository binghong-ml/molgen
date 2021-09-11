import torch
import torch.nn as nn

import math

from tqdm import tqdm

from model.vectorquantization import VectorQuantization
from data.util import allowable_features, node_feature_names, edge_feature_names

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
        return self.embedding(tokens) * math.sqrt(self.emb_size)

class EdgeLogitLayer(nn.Module):
    def __init__(self, emb_size, hidden_dim, vocab_size):
        super(EdgeLogitLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, vocab_size * self.hidden_dim)
        self.linear1 = nn.Linear(emb_size, vocab_size * self.hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out0 = self.linear0(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)
        out1 = self.linear1(x).view(batch_size, seq_len, self.vocab_size, self.hidden_dim)
        
        out0 = out0.permute(0, 2, 1, 3).reshape(-1, seq_len, self.hidden_dim)
        out1 = out1.permute(0, 2, 3, 1).reshape(-1, self.hidden_dim, seq_len)
        logits = self.scale * torch.bmm(out0, out1).view(batch_size, self.vocab_size, seq_len, seq_len)
        logits = logits.permute(0, 2, 3, 1)

        return logits

class BaseGenerator(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        logit_hidden_dim,
        vq_vocab_size,
    ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.pos_emb = AbsolutePositionalEncoding(emb_size)
        
        self.node_emb_dict = nn.ModuleDict(
            {key: TokenEmbedding(len(allowable_features[key]), emb_size) for key in node_feature_names}
            )
        self.edge_emb_dict = nn.ModuleDict(
            {key: TokenEmbedding(len(allowable_features[key]) + 1, nhead) for key in edge_feature_names}
            )

        #
        self.input_dropout = nn.Dropout(dropout)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        #
        decoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        decoder_norm = nn.LayerNorm(emb_size)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers, decoder_norm)

        #
        self.vq_layer = VectorQuantization(emb_size, vq_vocab_size)

        #
        self.node_generator_dict = nn.ModuleDict(
            {key: nn.Linear(emb_size, len(allowable_features[key])) for key in node_feature_names}
        )
        self.edge_generator_dict = nn.ModuleDict(
            {key: EdgeLogitLayer(emb_size, logit_hidden_dim, len(allowable_features[key]) + 1) for key in edge_feature_names}
        )

    def forward(self, batched_node_data, batched_edge_data):
        batch_size = batched_node_data[node_feature_names[0]].size(0)
        sequence_len = batched_node_data[node_feature_names[0]].size(1)
        batched_node_data = {key: batched_node_data[key].transpose(0, 1) for key in node_feature_names}
            
        #
        out = self.pos_emb(sequence_len)
        for key in node_feature_names:
            out = out + self.node_emb_dict[key](batched_node_data[key])

        out = self.input_dropout(out)

        #
        mask = 0.0
        for key in edge_feature_names:
            mask += self.edge_emb_dict[key](batched_edge_data[key])

        mask = mask.permute(0, 3, 1, 2)
        mask = mask.reshape(-1, sequence_len, sequence_len)
                
        #
        key_padding_mask = (batched_node_data[node_feature_names[0]] == 0).transpose(0, 1)

        out = self.encoder(out, mask, key_padding_mask)
        
        out = out.transpose(0, 1)
        out, _, vq_loss = self.vq_layer(out, key_padding_mask)
        out = out.transpose(0, 1)
        
        out = self.decoder(out, None, key_padding_mask)
        out = out.transpose(0, 1)
        
        node_logits = {key: self.node_generator_dict[key](out) for key in node_feature_names}
        edge_logits = {key: self.edge_generator_dict[key](out) for key in edge_feature_names}

        return node_logits, edge_logits, vq_loss

    def encode(self, batched_node_data, batched_edge_data):
        batch_size = batched_node_data[node_feature_names[0]].size(0)
        sequence_len = batched_node_data[node_feature_names[0]].size(1)
        batched_node_data = {key: batched_node_data[key].transpose(0, 1) for key in node_feature_names}
            
        #
        out = self.pos_emb(sequence_len)
        for key in node_feature_names:
            out = out + self.node_emb_dict[key](batched_node_data[key])

        out = self.input_dropout(out)

        #
        mask = 0.0
        for key in edge_feature_names:
            mask += self.edge_emb_dict[key](batched_edge_data[key])

        mask = mask.permute(0, 3, 1, 2)
        mask = mask.reshape(-1, sequence_len, sequence_len)
                
        #
        key_padding_mask = (batched_node_data[node_feature_names[0]] == 0).transpose(0, 1)

        out = self.encoder(out, mask, key_padding_mask)
        
        out = out.transpose(0, 1)
        out, out_ind, _ = self.vq_layer(out, key_padding_mask)
        
        return out_ind, key_padding_mask
        
    def decode(self, ind, key_padding_mask):
        out = self.vq_layer.compute_embedding(ind)
        out = out.transpose(0, 1)

        out = self.decoder(out, None, key_padding_mask)
        out = out.transpose(0, 1)
        
        node_logits = {key: self.node_generator_dict[key](out) for key in node_feature_names}
        edge_logits = {key: self.edge_generator_dict[key](out) for key in edge_feature_names}

        node_ids = {key: torch.argmax(node_logits[key], dim=-1) for key in node_feature_names}
        edge_ids = {key: torch.argmax(edge_logits[key], dim=-1) for key in edge_feature_names}

        return node_ids, edge_ids
        