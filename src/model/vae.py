import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from data.util import ALLOWABLE_FEATURES, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES, NODE_TARGET_NAMES, EDGE_TARGET_NAMES

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

class Encoder(nn.Module):
    def __init__(self, num_layers, emb_size,nhead, dim_feedforward):
        super(Encoder, self).__init__()
        #
        self.pos_emb = AbsolutePositionalEncoding(emb_size)
        
        self.node_emb_dict = nn.ModuleDict(
            {key: TokenEmbedding(len(ALLOWABLE_FEATURES[key]), emb_size) for key in NODE_FEATURE_NAMES}
            )
        self.edge_emb_dict = nn.ModuleDict(
            {key: TokenEmbedding(len(ALLOWABLE_FEATURES[key]), nhead) for key in EDGE_FEATURE_NAMES}
            )

        transformer_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout=0.0, activation="gelu")
        normalization_layer = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers, normalization_layer)

    def forward(self, batched_node_data, batched_edge_data):
        sequence_len = batched_node_data[NODE_FEATURE_NAMES[0]].size(1)
        batched_node_data = {key: batched_node_data[key].transpose(0, 1) for key in NODE_FEATURE_NAMES}
            
        #
        out = self.pos_emb(sequence_len)
        for key in NODE_FEATURE_NAMES:
            out = out + self.node_emb_dict[key](batched_node_data[key])

        #
        mask = 0.0
        for key in EDGE_FEATURE_NAMES:
            mask += self.edge_emb_dict[key](batched_edge_data[key])

        mask = mask.permute(0, 3, 1, 2)
        mask = mask.reshape(-1, sequence_len, sequence_len)
                
        #
        key_padding_mask = (batched_node_data[NODE_FEATURE_NAMES[0]] == 0).transpose(0, 1)

        out = self.transformer(out, mask, key_padding_mask)
        return out, key_padding_mask

class Decoder(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, dim_feedforward, logit_hidden_dim):
        super(Decoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout=0.0, activation="gelu")
        normalization_layer = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers, normalization_layer)

        self.node_generator_dict = nn.ModuleDict(
            {key: nn.Linear(emb_size, len(ALLOWABLE_FEATURES[key])) for key in NODE_TARGET_NAMES}
        )
        self.edge_generator_dict = nn.ModuleDict(
            {key: EdgeLogitLayer(emb_size, logit_hidden_dim, len(ALLOWABLE_FEATURES[key])) for key in EDGE_TARGET_NAMES}
        )

    def forward(self, batched_hidden_data, key_padding_mask):
        out = self.transformer(batched_hidden_data, None, key_padding_mask)
        out = out.transpose(0, 1)
        
        node_logits = {key: self.node_generator_dict[key](out) for key in NODE_TARGET_NAMES}
        edge_logits = {key: self.edge_generator_dict[key](out) for key in EDGE_TARGET_NAMES}

        return node_logits, edge_logits

class NormalCodeLayer(nn.Module):
    def __init__(self, emb_size, code_dim):
        super(NormalCodeLayer, self).__init__()
        self.code_dim = code_dim
        self.mu_layer = nn.Linear(emb_size, code_dim)
        self.logstd_layer = nn.Linear(emb_size, code_dim)
        self.latent_emb_layer = nn.Linear(code_dim, emb_size)

    def forward(self, batched_hidden_data, key_padding_mask):
        mu = self.mu_layer(batched_hidden_data)
        std = self.logstd_layer(batched_hidden_data).exp() + 1e-6
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = Normal(mu, std)
        z = q.rsample()

        out = self.latent_emb_layer(z)
        kl_loss = (q.log_prob(z) - p.log_prob(z))[~key_padding_mask].mean()
        
        return out, kl_loss
    
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        logit_hidden_dim,
        code_dim,
        num_nodes_list, 
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.code_dim = code_dim
        self.num_nodes_list = num_nodes_list

        self.encoder = Encoder(num_encoder_layers, emb_size, nhead, dim_feedforward)
        self.decoder = Decoder(num_decoder_layers, emb_size, nhead, dim_feedforward, logit_hidden_dim)
        self.code_layer = NormalCodeLayer(emb_size, code_dim)

    def forward(self, batched_node_data, batched_edge_data):
        out, key_padding_mask = self.encoder(batched_node_data, batched_edge_data)
        
        out = out.transpose(0, 1)
        out, reg_loss = self.code_layer(out, key_padding_mask)
        out = out.transpose(0, 1)
        
        node_logits, edge_logits = self.decoder(out, key_padding_mask)
        return node_logits, edge_logits, reg_loss

    def step(self, batched_node_data, batched_edge_data):
        statistics = dict()
        
        node_logits, edge_logits, reg_loss = self(batched_node_data, batched_edge_data)
        statistics["loss/reg"] = reg_loss

        correct_total = 1.0
        for key in NODE_TARGET_NAMES + EDGE_TARGET_NAMES:
            logits = node_logits[key] if key in node_logits else edge_logits[key]
            targets = batched_node_data[key] if key in batched_node_data else batched_edge_data[key]
            
            recon_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=ALLOWABLE_FEATURES[key].index('<pad>'),
                ) 
            
            preds = torch.argmax(logits, -1)
            correct = preds == targets
            correct[targets == 0] = True
            elem_acc = correct[targets != 0].float().mean()
            sequence_acc = correct.view(correct.size(0), -1).all(dim=1).float().mean()
            
            correct_total = correct_total * correct.view(correct.size(0), -1).all(dim=1).float()
            statistics[f"loss/{key}"] = recon_loss
            statistics[f"elem_acc/{key}"] = elem_acc 
            statistics[f"seq_acc/{key}"] = sequence_acc

        statistics["acc/total"] = correct_total.mean()

        return statistics

    def sample(self, num_samples, device):
        num_nodes_list = random.sample(self.num_nodes_list, k=num_samples)
        max_num_nodes = max(num_nodes_list)
        key_padding_mask = torch.ones(num_samples, max_num_nodes, dtype=torch.bool, device=device)
        for idx in range(num_samples):
            key_padding_mask[idx][:num_nodes_list[idx]] = False

        p = Normal(
            torch.zeros(num_samples, max_num_nodes, self.code_dim, device=device), 
            torch.ones(num_samples, max_num_nodes, self.code_dim, device=device)
            )
        z = p.sample()
        out = self.code_layer.latent_emb_layer(z)
        out = out.transpose(0, 1)

        node_logits, edge_logits = self.decoder(out, key_padding_mask)
        node_preds = {key: torch.argmax(node_logits[key], dim=-1) for key in node_logits}
        edge_preds = {key: torch.argmax(edge_logits[key], dim=-1) for key in edge_logits}

        return node_preds, edge_preds