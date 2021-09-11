import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import random
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.emb_size)

class SetGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        dropout,
        latent_dim,
    ):
        super(SetGenerator, self).__init__()
        self.nhead = nhead
        
        self.emb_layer = TokenEmbedding(vocab_size, emb_size)
        self.input_dropout = nn.Dropout(dropout)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.latent_dim = latent_dim
        self.mu_layer = nn.Linear(emb_size, latent_dim)
        self.logstd_layer = nn.Linear(emb_size, latent_dim)
        self.latent_emb_layer = nn.Linear(latent_dim, emb_size)

        #
        decoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        decoder_norm = nn.LayerNorm(emb_size)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.generator = nn.Linear(emb_size, vocab_size)

    def forward(self, batched_data):
        batched_data = batched_data.transpose(0, 1)
        #
        out = self.emb_layer(batched_data)
        out = self.input_dropout(out)
        
        #
        key_padding_mask = (batched_data == 0).transpose(0, 1)
        out = self.encoder(out, None, key_padding_mask)
        
        out = out.transpose(0, 1)
        mu = self.mu_layer(out)
        std = self.logstd_layer(out).exp() + 1e-6
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        out = self.latent_emb_layer(z)
        out = out.transpose(0, 1)
        
        out = self.decoder(out, None, key_padding_mask)
        out = out.transpose(0, 1)
        logits = self.generator(out)

        kl_loss = (q.log_prob(z) - p.log_prob(z))[~key_padding_mask].mean()
        
        return logits, kl_loss
    
    def update_num_latent_prior(self, num_latents):
        self.num_latents = num_latents
    
    def sample(self, batch_size, device):
        lengths = random.sample(self.num_latents, k=batch_size)
        max_len = max(lengths)
        key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for idx in range(batch_size):
            key_padding_mask[idx][:lengths[idx]] = False

        p = Normal(
            torch.zeros(batch_size, max_len, self.latent_dim, device=device), 
            torch.ones(batch_size, max_len, self.latent_dim, device=device)
            )
        z = p.sample()
        out = self.latent_emb_layer(z)
        out = out.transpose(0, 1)

        out = self.decoder(out, None, key_padding_mask)
        out = out.transpose(0, 1)
        logits = self.generator(out)

        return torch.argmax(logits, dim=-1), key_padding_mask
