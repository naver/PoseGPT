# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from argparse import ArgumentParser
import ipdb

""" VectorQuantizer code adapted from by https://github.com/CompVis/taming-transformers/taming/modules/vqvae/quantize.py"""

__all__ = ['VectorQuantizer']

def L2_efficient(x, y):
    return (x.pow(2).sum(1, keepdim=True) - 2 * x @ y + y.pow(2).sum(0, keepdim=True))

class EmaCodebookMeter:
    """Compute an estimate of centroid usage, using an EMA to track proportions """

    def __init__(self, codebook_size, ema_alpha=0.05):
        self.codebook_size = codebook_size
        self.bins = (torch.ones((self.codebook_size), requires_grad=False) / self.codebook_size).detach().cuda()
        self.ema_alpha = ema_alpha
        self.iters = 0

    def bincount(self, val, weights=None):
        norm = val.shape[0]
        weights = weights.reshape(-1) if weights is not None else None
        count = torch.bincount(val.reshape(-1), minlength=self.codebook_size,
                               weights=weights).detach()
        self.iters += 1
        return count / norm

    def load(self, bins):
        self.bins = torch.tensor(bins, requires_grad=False).detach().cuda()

    def update(self, val, weights=None, n=1):
        """ Count usage of each value in the codebook """
        count = self.bincount(val, weights=weights)
        alpha = max(self.ema_alpha, 1 / (self.iters + 1))
        self.bins = (1. - alpha) * self.bins + alpha * count

    def get_hist(self):
        return self.bins

class VectorQuantizer(nn.Module):
    """
    Code taken from https://github.com/CompVis/taming-transformers/
            blob/9d17ea64b820f7633ea6b8823e1f78729447cb57/taming/
            modules/vqvae/quantize.py#L213
    for handling input of shape [batch_size, seq_len, hid_dim]

    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, n_e, e_dim, beta,
                 nbooks=1, balance=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.nbooks = nbooks
        self.balance = balance

        assert n_e % nbooks == 0, "nb codebooks should divide nb centroids"
        self.n_e_i = n_e // nbooks

        embed_dims = (nbooks - 1) * [e_dim // nbooks] + \
                [e_dim - (nbooks - 1) * (e_dim // nbooks)]
        self.embed_dims = embed_dims

        self.embeddings = torch.nn.ModuleDict({str(i): nn.Embedding(self.n_e_i, d) for i, d in enumerate(embed_dims)})

        self.trackers = {}
        for i, e in self.embeddings.items():
            e.weight.data.uniform_(-1.0 / self.n_e_i, 1.0 / self.n_e_i)
            print(f"Codebook {i}: {list(e.weight.size())}")

            self.trackers[int(i)] = EmaCodebookMeter(self.n_e_i)

    def get_state(self):
        return {i: self.trackers[i].get_hist().cpu().data.numpy() for i in self.trackers.keys()}

    def load_state(self, bins):
        for i, b in bins.items():
            self.trackers[i].load(b)

    def get_hist(self, i):
        return self.trackers[i].get_hist()

    def reset(self, i):
        for i in self.trackers.keys():
            self.trackers = EmaCodebookMeter(self.embed_dims[int(i)])

    def track_assigment(self, emb_ind, i):
        self.trackers[i].update(emb_ind)

    def forward_one(self, z, i, weights=None):
        bsize = self.e_dim // self.nbooks
        e_dim = bsize  if i < self.nbooks - 1 else self.e_dim - (self.nbooks - 1) * bsize

        z_flattened = z.view(-1, e_dim)
        dist = L2_efficient(z_flattened, self.embeddings[str(i)].weight.t())

        if self.balance and weights is not None:
            #weights = (proportions * self.n_embed).unsqueeze(0)
            wdist = dist * weights.unsqueeze(0)
            dist = -torch.nn.functional.softmax(-wdist, 1)

        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e_i).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if self.training:
            self.track_assigment(min_encoding_indices.detach(), i)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings[str(i)].weight).view(z.shape)
        
        #min_encoding_indices.view(z.shape)
        return z_q, min_encoding_indices.view(z.shape[:-1] + (1,))

    def forward(self, z, p=1.0):
        assert z.size(2) == self.e_dim
        zs = torch.split(z, z.size(2) // len(self.embeddings), dim=-1)
        zq_i = [self.forward_one(z, i, self.get_hist(i)) for i, z in enumerate(zs)]
        z_q, min_encoding_indices = [torch.cat([e[i] for e in zq_i], dim=-1) for i in [0,1]]

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2, dim=-1) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2, dim=-1)

        if p != 1.0:
            # Proba of being quantized.
            quant_mask = torch.bernoulli(p * torch.ones_like(z)).float()
            z_q = quant_mask * z_q + (1 - quant_mask) * z

        # preserve gradients
        z_q = z + (z_q - z).detach()
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, eos_mask=None):
        """
        Args:
            - indices: [batch_size,seq_len]
        Return:
            - z_q: [batch_size,seq_len,e_dim]
        """
        # This is a hack, but it enables us to keep the '-1' index solely in the gpt
        embds = [self.embeddings[str(i)](e.squeeze(-1)) for i, e in enumerate(torch.split(indices, 1, dim=-1))]
        return torch.cat(embds, dim=-1)

if __name__ == "__main__":
    device = torch.device('cuda')
    model = VectorQuantizer(n_e=1024, e_dim=128, beta=0.25, nbooks=16, balance=False).to(device)
    x = torch.randn(32, 64, 128).to(device)

    # two forwards
    z_q1, _, indices1 = model(x)
    z_q2, _, indices2 = model(x)
    print((z_q1 - z_q2).abs().sum())

    # from indices to codebook entry
    z_q3 = model.get_codebook_entry(indices1)
    print((z_q1 - z_q3).abs().sum())
