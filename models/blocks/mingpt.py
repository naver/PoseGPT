"""
Based on Andrej karpathy's minGPT (https://github.com/karpathy/minGPT/).
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
from models.blocks.attention import CausalSelfAttention

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

def pad(z):
    """ Pad by one for auto-regression: first value is predicted from nothing. """
    b, _, c = z.shape
    pad = torch.zeros((b, 1, c)).long().to(z.device)
    return torch.cat((pad, z), dim=1)

class Block(nn.Module):
    def __init__(self, config, in_factor=None, out_factor=None):
        super().__init__()
        in_dim = config.n_embd * (in_factor if in_factor is not None else 1)
        out_dim = config.n_embd * (out_factor if out_factor is not None else 1)
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, in_dim=in_dim if in_factor is not None else None)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, out_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False, valid_mask=None, in_residual=True, out_residual=True):

        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past,
                                  valid_mask=valid_mask)
        x = x + attn if in_residual else attn
        x = x + self.mlp(self.ln2(x)) if out_residual else self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x




