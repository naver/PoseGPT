# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
import logging
import torch
from torch import nn
from argparse import ArgumentParser
from .blocks.mingpt import Block
from .blocks.convolutions import Masked_conv, Masked_up_conv
from .quantize import VectorQuantizer
import roma

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, type='sine_frozen', max_len=1024, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        if 'sine' in type:
            rest = dim % 2
            pe = torch.zeros(max_len, dim+rest)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim+rest, 2).float() * (-math.log(10000.0) / (dim+rest)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:,:dim]
            pe = pe.unsqueeze(0)  # [1,t,d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, dim))  # [1,t,d]
            self.register_buffer('pe', pe)
        else:
            raise NameError

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x

class Encoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class Stack(nn.Module):
    """ A stack of transformer blocks.
        Used to implement a U-net structure """

    def __init__(self, block_size, n_layer=12, n_head=8, n_embd=256,
                 dropout=0.1, causal=False, down=1, up=1,
                 pos_type='sine_frozen', sample_method='conv',
                 pos_all=False):
        super().__init__()
        config = Encoder_Config(block_size, n_embd=n_embd, n_layer=n_layer,
                                n_head=n_head, dropout=dropout, causal=causal)
        self.drop = nn.Dropout(dropout)
        assert down == 1 or up == 1, "Unexpected combination"
        assert down in [1, 2] and up in [1, 2], "Not implemented"
        assert sample_method in ['cat', 'conv'], "Unknown sampling method"
        cat_down, slice_up = (down, up) if sample_method == 'cat' else (1, 1)
        self.cat_down, self.slice_up = cat_down, slice_up
        self.pos_all = pos_all
        self.blocks = nn.ModuleList([])
        self.pos = nn.ModuleList([])
        for i in range(config.n_layer):
            # Inside Block, standard transformer stuff happens.
            self.blocks.append(Block(config,
                                     in_factor=cat_down if i == 0 and cat_down > 1 else None,
                                     out_factor=slice_up if i == config.n_layer - 1 and slice_up > 1 else None))
            in_dim = config.n_embd * (cat_down if i == 0 and cat_down > 1 else 1)
            if pos_all or i == 0:
                self.pos.append(PositionalEncoding(dim=in_dim, max_len=block_size, type=pos_type))
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.down_conv, self.up_conv = None, None
        if sample_method == 'conv':
            if down == 2:
                self.down_conv = Masked_conv(config.n_embd, config.n_embd,
                                             masked=causal, pool_size=down, pool_type='max')
            elif up == 2:
                self.up_conv = Masked_up_conv(config.n_embd, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x=None, z=None, mask=None, **kwargs):
        """
        Args:
            - idx: [batch_size, seq_len, n_codebook]
            - actions_emb: [batch_size, 1, emb_dim]
            - seqLen_emb: [batch_size, 1, emb_dim]
        Return:
            - logits: [batch_size, seq_len, n_codebook, codebook_size]
        """
        mask = mask.bool()
        assert (x is None)^(z is None), "Only x or z as input"
        x = x if x is not None else z

        t = x.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.cat_down > 1:
            if self.cat_down != 2:
                raise NotImplementedError
            else:
                x = rearrange(x, 'b (t t2) c -> b t (t2 c)', t2=2)
                mask = rearrange(mask, 'b (t t2)-> b t t2', t2=2)[:, :, 0]

        if self.down_conv is not None:
            x, mask = self.down_conv(x, mask)

        x = self.drop(x)
        for i in range(len(self.blocks)):
            x = self.pos[i](x) if (i == 0 or self.pos_all) else x
            x = self.blocks[i](x, valid_mask=mask,
                               in_residual=not(i == 0 and self.cat_down > 1),
                               out_residual=not(i == (len(self.blocks) - 1) and self.slice_up > 1))
        if self.slice_up > 1:
            x = rearrange(x, 'b t (t2 c) -> b (t t2) c', t2=2)

        if self.up_conv is not None:
            x, mask = self.up_conv(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, mask

class TransformerAutoEncoder(nn.Module):
    """
    Model composed of an encoder and a decoder.
    """
    # NOTE This is an abstract class for us
    # as we are not interested in vanilla autoencoders 
    # with low dimensionality bottlenecks, so it does not implement forward().
    def __init__(self, *, in_dim=1024, n_layers=6,
                 hid_dim=384, heads=4,
                 dropout=0.1, e_dim=256, block_size=2048,
                 causal_encoder=False, causal_decoder=False, mixin=False,
                 pos_type='sine_frozen', pos_all=False,
                 sample_method='conv', **kwargs):
        super().__init__()

        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        if len(hid_dim) != 1:
            raise NotImplementedError("Does not handle per-layer channel specification.")

        if not isinstance(n_layers, list):
            n_layers = [n_layers]

        # Constrols masking of  attention in encoder / decoder.
        self.causal_decoder = causal_decoder
        self.causal_encoder = causal_encoder

        n_embd = hid_dim[0]
        self.in_dim = in_dim
        self.emb = nn.Linear(in_dim, n_embd)

        if not len(n_layers) > 1:
            raise ValueError("With the new convention, this is probably" + \
                    "a mistake, unless you want to stay at full time resolution for the whole network.")
        # Build the encoder; basic brick is a 'Stack object'.
        self.encoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size,
                   n_layer=n_layers[i],
                   n_head=heads,
                   n_embd=n_embd,
                   dropout=dropout,
                   causal=causal_encoder,
                   down=2 if i else 1, pos_type=pos_type,
                   pos_all=pos_all, sample_method=sample_method) for i in range(len(n_layers))
            if n_layers[i]])

        # project features (hid) to latent variable dimensions (before going through bottleneck)
        # and then z to hid
        self.emb_in, self.emb_out = n_embd, e_dim
        self.quant_emb = nn.Linear(n_embd, e_dim)
        self.encoder_dim = e_dim
        self.post_quant_emb = nn.Linear(e_dim, n_embd)

        # Build the decoder
        self.decoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[i],
                   n_head=heads, n_embd=n_embd, causal=causal_decoder,
                   up=2 if i else 1, pos_type=pos_type, pos_all=pos_all) for i in list(range(len(n_layers)))[::-1]
                   if n_layers[i]])
        dim = n_embd
        # Final head to predict body and root paramaters
        self.reg_body = nn.Sequential(nn.Linear(dim, dim),
                                 nn.ReLU(), nn.Linear(dim, self.in_dim - 3))
        self.reg_root = nn.Sequential(nn.Linear(dim, dim),
                                 nn.ReLU(), nn.Linear(dim, 3))
    
    def encoder(self, return_mask=False, *args, **kwargs):
        """ Calls each encoder stack sequentially """
        o, m = self.encoder_stacks[0](*args, **kwargs)
        for i in range(1, len(self.encoder_stacks)):
            o, m = self.encoder_stacks[i](x=o, mask=m)
        return o, m if return_mask else o

    def decoder(self, return_mask=False, *args, **kwargs):
        """ Calls each decoder stack sequentially """
        o, m = self.decoder_stacks[0](*args, **kwargs)
        for i in range(1, len(self.encoder_stacks)):
            o, m = self.decoder_stacks[i](x=o, mask=m)
        return (o, m) if return_mask else o

    def regressor(self, x):
        return self.reg_body(x), self.reg_root(x)
    def prepare_batch(self, x, *args,  **kwargs):
        return x


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        ddict = {'hid_dim': 384, 'n_layers': [0, 6], 'heads': 4,
                 'dropout': 0., 'pos_type': 'sine_frozen',
                 'pos_all': 1, 'sample_method': 'conv'}
        parser.add_argument("--hid_dim", type=int, nargs='+', default=ddict['hid_dim'])
        parser.add_argument("--n_layers", type=int, nargs='+', default=ddict['n_layers'])
        parser.add_argument("--heads", type=int, default=ddict['heads'])
        parser.add_argument("--dropout", type=float, default=ddict['dropout'])
        parser.add_argument("--pos_type", type=str, default=ddict['pos_type'])
        parser.add_argument("--pos_all", type=int, default=ddict['pos_all'])
        parser.add_argument("--sample_method", type=str, default=ddict['sample_method'])
        return parser

class TransformerVQVAE(TransformerAutoEncoder):
    """
    Adds a quantization bottleneck to TransformerAutoEncoder.
    """
    def __init__(self, *, in_dim=1024,
                 # Arguments related to model structure (passed to parent class) 
                 n_layers=[0, 6], hid_dim=384, heads=4, dropout=0.1, causal_encoder=False, causal_decoder=False,
                 # Arguments related to quantization.
                 n_codebook=1, n_e=128, e_dim=256, beta=1.,
                 **kwargs):
        super().__init__(**{'n_layers': n_layers, 'hid_dim': hid_dim, 'e_dim': e_dim,
            'in_dim': in_dim, 'heads': heads, 'dropout': dropout, 'causal_encoder': causal_encoder,
            'causal_decoder': causal_decoder}, **kwargs)
        assert e_dim % n_codebook == 0
        assert n_e % n_codebook == 0
        self.seq_len = kwargs['seq_len']
        #self.factor = kwargs['factor']

        self.one_codebook_size = n_e // n_codebook

        targets = ['body', 'root', 'trans', 'vert', 'fast_vert']
        self.log_sigmas = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(1)) for k in targets})
        assert not any([a is None for a in [n_e, e_dim, beta]]), "Missing arguments"
        self.quantizer = VectorQuantizer(n_e=n_e, e_dim=e_dim,
                                         beta=beta, nbooks=n_codebook)

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
        """ Forward the encoder, return the quantized latent variables. """
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
        """ Take quantized indices and forward the decoder. """
        assert valid is not None, "With the new implementation, safer to explicitely set valid"
        z_q = self.quantizer.get_codebook_entry(indices)
        hid = self.post_quant_emb(z_q)
        y, valid = self.forward_decoder(z=hid, mask=valid, return_mask=True)
        batch_size, seq_len, *_ = y.size()
        rotmat, trans = self.regressor(y)
        rotmat = rotmat.reshape(batch_size, seq_len, -1, 3, 2)
        rotmat = roma.special_gramschmidt(rotmat)
        return (rotmat, trans), valid

    def quantize(self, z, valid=None, p=1.0):
        z_q, loss, indices = self.quantizer(z, p=p)

        if valid is not None:
            loss = torch.sum(loss * valid) / torch.sum(valid)
            valid = valid.unsqueeze(-1)
            indices = (indices) * valid + (1 - valid) * -1 * torch.ones_like(indices)
            z_q = z_q * valid
        return z_q, loss, indices

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(TransformerVQVAE, TransformerVQVAE).add_model_specific_args(parent_parser)
        #parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_codebook", type=int, default=1)
        parser.add_argument("--balance", type=int, default=0, choices=[0, 1])
        parser.add_argument("--n_e", type=int, default=512)
        parser.add_argument("--e_dim", type=int, default=128)
        parser.add_argument("--beta", type=float, default=0.25)
        parser.add_argument("--quant_min_prop", type=float, default=1.0)
        parser.add_argument("--quant_epoch_max", type=int, default=20)
        return parser

class CausalVQVAE(TransformerVQVAE):
    """ Masked transformer - Quantization - Masked Transformer """
    def __init__(self, **kwargs):
        super().__init__(causal_encoder=True, causal_decoder=False, **kwargs)

class OnlineVQVAE(TransformerVQVAE):
    """ Masked transformer - Quantization - Transformer """
    def __init__(self, **kwargs):
        super().__init__(causal_encoder=True, causal_decoder=True, **kwargs)

class OfflineVQVAE(TransformerVQVAE):
    """ Transformer - Quantization - Transformer """
    def __init__(self, **kwargs):
        super().__init__(causal_encoder=False, causal_decoder=False, **kwargs)
