# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
from tqdm import tqdm
import os
import pickle as pkl
import torch
from torch import nn
import copy

from argparse import ArgumentParser
from utils.constants import NUM_JOINTS_SMPLH, NUM_JOINTS_SMPL, SMPL_MEAN_PARAMS
from .blocks.vqvae_v1.blocks import Encoder, Decoder
from .quantize import VectorQuantizer
import numpy as np
import torch.nn.functional as F
import roma
import sys
from einops import rearrange
from utils.body_model import get_trans
from utils.variable_length import valid_concat_rot_trans
#from classiy import Autoreg_classifier, ActorMotionDiscriminator
from utils.variable_length import repeat_last_valid
from utils.fid import calculate_activation_statistics, calculate_frechet_distance
from utils.ae_utils import red
#from evaluate import build_classifier, forward_classifier, get_real_fid_path, 
# from evaluate import compute_fid
from functools import partial
import ipdb

class VQVAE(nn.Module):
    """ Abstract class for VQVAEs. """
    def __init__(self, *, n_e, n_codebook, e_dim, beta, balance, mixin=False, **kwargs):
        if not mixin:
            super(VQVAE, self).__init__()
        assert e_dim % n_codebook == 0
        assert n_e % n_codebook == 0
        self.seq_len = kwargs['seq_len']
        self.factor = kwargs['factor']

        self.one_codebook_size = n_e // n_codebook

        targets = ['body', 'root', 'trans', 'vert', 'fast_vert']
        self.log_sigmas = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(1)) for k in targets})
        assert not any([a is None for a in [n_e, e_dim, beta]]), "Missing arguments"
        self.quantizer = VectorQuantizer(n_e=n_e,
                                         e_dim=e_dim,
                                         beta=beta,
                                         nbooks=n_codebook,
                                         balance=balance)

    def prepare_batch(self, x, mask=None):
        """ Put x in the format expected by the encoder """
        raise NotImplementedError("This needs to be implemented")

    def forward_encoder(self, x, mask):
        """"
        Run the forward pass of the encoder
        """
        batch_size, seq_len, *_ = x.size()
        x = self.emb(x)
        hid, mask = self.encoder(x=x, mask=mask, return_mask=True)
        return hid, mask

    def forward_decoder(self, z, mask, return_mask=False):
        """"
        Run the forward pass of the decoder
        """
        seq_len_hid = z.shape[1]
        return self.decoder(z=z, mask=mask, return_mask=return_mask)

    def forward(self, *, x, y, valid, quant_prop=1.0, **kwargs):
        mask = valid
        batch_size, seq_len, *_ = x.size()
        x = self.prepare_batch(x, mask)
        
        hid, mask_ = self.forward_encoder(x=x, mask=mask)

        z = self.quant_emb(hid)

        if mask_ is None:
            mask_ = F.interpolate(mask.float().reshape(batch_size, 1, seq_len, 1),
                                  size=[z.size(1), 1], mode='bilinear', align_corners=True)
            mask_ = mask_.long().reshape(batch_size, -1)
        else:
            mask_ = mask_.float().long()

        z_q, z_loss, indices = self.quantize(z, mask_, p=quant_prop)

        hid = self.post_quant_emb(z_q) # this one is i.i.d

        y = self.forward_decoder(z=hid, mask=mask_)

        rotmat, trans = self.regressor(y)
    
        rotmat = rotmat.reshape(batch_size, seq_len, -1, 3, 2)

        rotmat = roma.special_gramschmidt(rotmat)

        kl = math.log(self.one_codebook_size) * torch.ones_like(indices, requires_grad=False)

        return (rotmat, trans), {'quant_loss': z_loss, 'kl': kl, 'kl_valid': mask_}, indices

    def forward_latents(self, x, mask, return_indices=False, do_pdb=False, return_mask=False, *args, **kwargs):
        batch_size, seq_len, *_ = x.size()
        x = self.prepare_batch(x, mask)
        hid, mask_ = self.forward_encoder(x=x, mask=mask)
        z = self.quant_emb(hid)
        if mask_ is None:
            mask_ = F.interpolate(mask.float().reshape(batch_size, 1, seq_len, 1),
                                  size=[z.size(1), 1], mode='bilinear', align_corners=True)
            mask_ = mask_.long().reshape(batch_size, -1)
        else:
            mask_ = mask_.float().long()

        # indices can contain -1 values
        z_q, z_loss, indices = self.quantize(z, mask_)

        if return_indices and return_mask:
            return z_q, indices, mask_
        if return_indices:
            return z_q, indices
        if return_mask:
            return z_q, mask_

        return z_q

    def forward_from_indices(self, indices, valid=None):
        assert valid is not None, "With the new implementation, safer to explicitely set valid"
        z_q = self.quantizer.get_codebook_entry(indices)
        hid = self.post_quant_emb(z_q)
        y, valid = self.forward_decoder(z=hid, mask=valid, return_mask=True)
        batch_size, seq_len, *_ = y.size()
        rotmat, trans = self.regressor(y)
        rotmat = rotmat.reshape(batch_size, seq_len, -1, 3, 2)
        rotmat = roma.special_gramschmidt(rotmat)
        return (rotmat, trans), valid

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_codebook", type=int, default=1)
        parser.add_argument("--balance", type=int, default=0, choices=[0, 1])
        parser.add_argument("--n_e", type=int, default=512)
        parser.add_argument("--e_dim", type=int, default=128)
        parser.add_argument("--beta", type=float, default=0.25)
        parser.add_argument("--quant_min_prop", type=float, default=1.0)
        parser.add_argument("--quant_epoch_max", type=int, default=20)
        return parser


    def quantize(self, z, valid=None, p=1.0):
        z_q, loss, indices = self.quantizer(z, p=p)

        if valid is not None:
            loss = torch.sum(loss * valid) / torch.sum(valid)
            valid = valid.unsqueeze(-1)
            indices = (indices) * valid + (1 - valid) * -1 * torch.ones_like(indices)
            z_q = z_q * valid
        return z_q, loss, indices

