import torch
import torch.nn as nn
import math
from einops import rearrange

""" Slightly adapted from  https://github.com/karpathy/minGPT/blob/master/mingpt/model.py """

class CausalSelfAttention(nn.Module):
    """
    Self-attention, possibly causal.
    """

    def __init__(self, config, in_dim=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.causal = config.causal
        # key, query, value projections for all heads
        if in_dim is None:
            in_dim = config.n_embd
        self.kdim = config.n_embd
        self.key = nn.Linear(in_dim, config.n_embd)
        self.query = nn.Linear(in_dim, config.n_embd)
        self.value = nn.Linear(in_dim, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            mask = torch.tril(torch.ones(config.block_size,
                                         config.block_size))
            if hasattr(config, "n_unmasked"):
                mask[:config.n_unmasked, :config.n_unmasked] = 1
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, valid_mask=None):
        B, T, C = x.size()

        C = self.kdim
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if layer_past is None and self.causal:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        if valid_mask is not None:
            valid_mask = rearrange(valid_mask, 'b j -> b () () j')
            att = att.masked_fill_(~valid_mask, float('-inf'))

        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


