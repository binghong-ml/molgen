import torch
from torch import nn
import torch.nn.functional as F

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantization(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, mask):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.argmin(dim=1)
        
        quantize = F.embedding(embed_ind.view(*input.shape[:-1]), self.embed.transpose(0, 1))
        quantize = input + (quantize - input).detach()        
        loss = F.mse_loss(quantize.detach()[~mask], input[~mask])
        
        if self.training:
            flatten_mask = mask.view(-1)
            embed_onehot = F.one_hot(embed_ind[~flatten_mask], self.n_embed).type(torch.float)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            
            embed_sum = flatten[~flatten_mask, :].transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        return quantize, loss
    
    def compute_embedding_index(self):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.argmin(dim=1).view(*input.shape[:-1])
        return embed_ind        

    def compute_embedding(self, embed_ind):
        return F.embedding(embed_ind, self.embed.transpose(0, 1))