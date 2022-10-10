# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
from tqdm import tqdm
import torch
import pickle as pkl
from torch import nn
import roma
#from models.blocks.mingpt import GPT
from argparse import ArgumentParser
#from utils.ae_utils import freeze
from torch.nn import functional as F
from einops import rearrange
from utils.body_model import get_trans
from utils.variable_length import valid_concat_rot_trans, repeat_last_valid
from utils.ae_utils import red
import math
#from dataset.preprocessing.babel import get_glove_embeddings
from models.blocks.mingpt import Block, GPTConfig
from models.blocks.sampling import sample_from_logits as _sample_from_logits
from functools import partial

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def pad(z):
    """ Pad by one for auto-regression: first value is predicted from nothing. """
    b, _, c = z.shape
    pad = torch.zeros((b, 1, c)).long().to(z.device)
    return torch.cat((pad, z), dim=1)

class GPT(nn.Module):
    """  GPT like model, adapted to deal with human motion as input;
        context size is controlled with block_size .
    """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 n_codebook=1, embed_every_step=False, embed_seqlen=False, concat_emb=False,
                 head_type='fc_wo_bias', pos_emb='scratch', class_conditional=True,
                 causal=True, gen_eos=False):
        super().__init__()
        self.class_conditional = class_conditional
        self.gen_eos = gen_eos
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, causal=causal)
        # input embedding 
        self.tok_emb = nn.Embedding(config.vocab_size + n_codebook, config.n_embd // n_codebook)  # split the embedding by n_codebook

        # positional embedding
        self.set_pos_embedding(pos_emb, config)

        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer torso of the model (stack of attention - mlp blocks)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.set_output_head(n_codebook, head_type, config)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        self.embed_every_step = embed_every_step
        self.embed_seqlen = embed_seqlen
        self.concat_emb = concat_emb
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        
        if self.concat_emb:
            emb_count = 2 + int(class_conditional) + int(embed_seqlen)
        else:
            emb_count = 3 + int(embed_seqlen)
        self.fc_emb = nn.Linear((emb_count) * config.n_embd, config.n_embd) if embed_every_step else None

    def set_output_head(self, n_codebook, head_type, config):
        self.list_head = nn.ModuleList()
        for _ in range(n_codebook):
            nb_out = config.vocab_size // n_codebook + (1 if self.gen_eos else 0) # +1 is for end of sequence token
            if head_type == 'fc_wo_bias':
                self.list_head.append(nn.Linear(config.n_embd, nb_out, bias=False))
            elif head_type == 'mlp':
                print("Using MLP heads")
                self.list_head.append(
                    nn.Sequential(nn.Linear(config.n_embd, config.n_embd),
                    nn.ReLU(),
                    nn.Linear(config.n_embd, nb_out))
                )
            else:
                raise NameError

    def set_pos_embedding(self, pos_emb, config):
        # positional embedding  (several types)
        if 'scratch' in pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        elif 'sine' in pos_emb:
            n_seqlens, gpt_nembd = config.block_size, config.n_embd
            pe = torch.zeros(n_seqlens, gpt_nembd)
            position = torch.arange(0, n_seqlens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, gpt_nembd, 2).float() * (-math.log(10000.0) / gpt_nembd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_emb = nn.Parameter(pe.unsqueeze(0))
        else:
            raise NameError("Unknown embedding")
        if 'frozen' in pos_emb:
            self.pos_emb.requires_grad = False # default at True


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

    def get_token_embedding_shape(self, batch_size, n_codebook, idx):
        # Create a dummy tensor with 1 as temporal dimension to get the correct dimensions for token embedding.
        token_embeddings = self.tok_emb(torch.ones((batch_size * n_codebook), requires_grad=False).to(idx.device).long())
        token_embeddings = token_embeddings.reshape(batch_size, 1, n_codebook, -1)
        token_embeddings = token_embeddings.flatten(2)[:, :0, :]
        return token_embeddings

    def cat_indices(self, idx):
        idx = torch.cat([x + (i * self.config.vocab_size // idx.shape[-1])
                          for i, x in enumerate(idx.split(1, dim=-1))], dim=-1)
        return idx

    def forward(self, idx, actions_emb=None, seqlens_emb=None):
        """
        Args:
            - idx: [batch_size, seq_len, n_codebook]
            - actions_emb: [batch_size, 1, emb_dim]
            - seqLen_emb: [batch_size, 1, emb_dim]
        Return:
            - logits: [batch_size, seq_len, n_codebook, codebook_size]
        """
        assert not (self.embed_seqlen and not self.embed_every_step), "Seqlength embedding is only coded with embed_every_step"
        assert not (self.embed_seqlen and seqlens_emb is None), "This model needs to be conditioned on sequence length"
        batch_size, seq_len, n_codebook = idx.size()

        # forward the GPT model
        idx = self.cat_indices(idx)

        if seq_len != 0:
            token_embeddings = self.tok_emb(idx.flatten(1))  # each index maps to a (learnable) vector
            token_embeddings = token_embeddings.reshape(batch_size, seq_len, n_codebook, -1)
            token_embeddings = token_embeddings.flatten(2)  # concat codebook embedding TODO @fbaradel add args
        else:
            token_embeddings = self.get_token_embedding_shape(batch_size, n_codebook, idx)

        # prepend explicit embeddings
        if actions_emb is not None and not self.embed_every_step:
            token_embeddings = torch.cat((actions_emb, token_embeddings), dim=1)
        else:
            token_embeddings = pad(token_embeddings)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        list_emb = [token_embeddings, position_embeddings.repeat(batch_size, 1, 1)]
        if actions_emb is not None:
            list_emb += [actions_emb.repeat(1, t, 1) if actions_emb.shape[1] == 1 else pad(actions_emb)[:,:t]] if self.embed_every_step else []
        list_emb += [seqlens_emb.repeat(1, t, 1)] if self.embed_seqlen and self.embed_every_step else []
        
        if self.concat_emb:
            x = self.fc_emb(torch.cat(list_emb, -1))
        else:
            x = torch.stack(list_emb, -1).sum(-1)

        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = torch.stack([head(x) for head in self.list_head], 2) # [batch_size, seq_len, n_codebook, codebook_size]
        return logits

class poseGPT(nn.Module):
    """
    The full model:
            - has a causal encoder and a quantization bottleneck to encode motion into sequences of indices
            - a GPT to model that seqence (predict next token given past or sample from it)
            - A decoder to decode it into motion.
    """
    def __init__(self, *, n_e, gpt_nembd, gpt_blocksize, gpt_nhead, gpt_nlayer, vqvae, causal_gpt, **kwargs):
        super(poseGPT, self).__init__()
        gpt_config = {'vocab_size': n_e,
                      'block_size': gpt_blocksize,
                      'n_layer': gpt_nlayer,
                      'n_head': gpt_nhead,
                      'n_embd': gpt_nembd,
                      'n_codebook': kwargs['n_codebook'],
                      'embd_pdrop': kwargs['gpt_embd_pdrop'],
                      'resid_pdrop': kwargs['gpt_resid_pdrop'],
                      'attn_pdrop': kwargs['gpt_attn_pdrop'],
                      'embed_every_step': kwargs['embed_every_step'],
                      'embed_seqlen': kwargs['seqlen_conditional'],
                      'concat_emb': kwargs['concat_emb'],
                      'head_type': kwargs['head_type'],
                      'pos_emb': kwargs['pos_emb'],
                      'causal': causal_gpt,
                      'class_conditional': kwargs['class_conditional'],
                      'gen_eos': kwargs['gen_eos']}
        #self.continuous = continuous
        self.causal = causal_gpt
        self.gen_eos = kwargs['gen_eos']
        self.gpt = GPT(**gpt_config)
        self.class_conditional = kwargs['class_conditional']
        self.seqlen_conditional = kwargs['seqlen_conditional']

        # The auto-encoder is frozen
        self.vqvae = vqvae
        self.vqvae.eval()
        freeze(self.vqvae)

        self.vocab_size = n_e
        self.autoreg_pq = kwargs['autoreg_pq']

        if self.autoreg_pq:
            self.set_autoreg_head(gpt_nembd, kwargs)

        n_actions = 64
        self.set_action_embedding(n_actions, gpt_nembd, kwargs)
        n_seqlens = 1024
        self.set_seqlen_embedding(n_seqlens, gpt_nembd, kwargs)
        self.factor = kwargs['factor']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--gpt_blocksize", type=int, default=2048)
        parser.add_argument("--gpt_nlayer", type=int, default=12)
        parser.add_argument("--gpt_nhead", type=int, default=8)
        parser.add_argument("--gpt_nembd", type=int, default=256)
        parser.add_argument("--gpt_embd_pdrop", type=float, default=0.)
        parser.add_argument("--gpt_resid_pdrop", type=float, default=0.)
        parser.add_argument("--gpt_attn_pdrop", type=float, default=0.)
        parser.add_argument("--embed_every_step", type=int, default=0, choices=[0,1])
        parser.add_argument("--autoreg_pq", type=int, default=0, choices=[0,1])
        parser.add_argument("--causal_gpt", type=int, default=1, choices=[0,1])
        #parser.add_argument("--head_type", type=str, default='fc_wo_bias', choices=['fc_wo_bias', 'mlp'])
        return parser


    def indices_to_embeddings(self, index, zshape=None):
        quant_z = self.vqvae.quantizer.get_codebook_entry(index)  # , shape=bhwc
        return quant_z

    def forward_from_indices(self, indices, eos, return_valid=False):
        """ Wrapper around vq, which handles the end of sequence token.
        If eos = 0, we substract 1 (eos becomes -1); then we mask all -1 using valid."""
        assert eos in [0, -1], 'Invalid eos value'
        # As soon as one of the elements is the end of sequence token, we consider that it is true for the rest.
        valid = (1. - (torch.cumsum(((indices == eos).sum(-1) > 0).float(), dim=-1) > 0).float()).int()
        if eos == 0:
            # We substract 1 from all indices (to remove eos_token), 
            indices -= 1
        # We replace eos_token by 0s (it will be masked in the decoder using valid)
        indices = valid.unsqueeze(-1) * indices + (1 - valid.unsqueeze(-1)) * torch.zeros_like(indices)
        result, valid = self.vqvae.forward_from_indices(indices, valid=valid)
        if return_valid:
            return result, valid
        return result

    def forward_gpt(self, indices, actions_emb, seqlens_emb):
        """ Handles the end of sequence token,
        then forwards the self.gpt model, then possibly adds the
        auto-regression on product quantization elements. """
        indices = indices + 1
        # Cut off the last value: it has observed the full input and cannot be used.
        logits = self.gpt(indices, actions_emb, seqlens_emb)[:, :-1, ...] # bs, t, K, n_e_i
        if self.autoreg_pq:
            t = logits.shape[1]
            idx_embed = self.gpt.tok_emb(
                    self.gpt.cat_indices(indices))
            idx_embed = rearrange(idx_embed, 'bs t K c -> (bs t) K c')
            logits = rearrange(logits, 'bs t K n_e_i -> (bs t) K n_e_i')
            idx_embed, logits = [x.split(1, dim=-2) for x in [idx_embed, logits]]
            outputs = [logits[0]]
            for i in range(1, len(idx_embed)):
                autoreg_inputs = torch.cat([*idx_embed[:i], *logits[i:]], dim=-1)
                outputs.append(self.autoreg_head[str(i)](autoreg_inputs))
            logits = rearrange(torch.cat(outputs, dim=-2), '(bs t) K c -> bs t K c', t=t)
        return logits

    def set_action_embedding(self, n_actions, gpt_nembd, kwargs):
        """ Define stems for action embeddings """
        self.action_emb_type = kwargs['action_emb']
        if self.action_emb_type == 'scratch':
            self.action_emb = nn.Embedding(n_actions, gpt_nembd)
        elif 'glove' in self.action_emb_type:
            if 'babel' in kwargs['train_data_dir']:
                dataset = 'babel'
            else:
                raise NotImplementedError("Unknown action_embed_type")
            glove_type='glove.6B.50d.txt'
            print(f"Loading pretrained GloVe embeddings: {glove_type}...")
            self.vocab_npa, self.embs_npa, actionId2WordIds, self.word2ActionId = get_glove_embeddings(glove_type=glove_type, dataset=dataset)
            self.action_emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embs_npa).float())
            self.action_emb.weight.requires_grad = False # frozen as default
            self.actionId2WordIds = {k: torch.Tensor(v).long() for k, v in actionId2WordIds.items()}
            self.action_emb_post = nn.Linear(self.embs_npa.shape[1], gpt_nembd, bias=False)
            if 'ft' in self.action_emb_type:
                self.action_emb.weight.requires_grad = True # finetune

    def set_seqlen_embedding(self, n_seqlens, gpt_nembd, kwargs):
        " Define stems for sequence length embeddings "
        self.sample_eos_force = kwargs['eos_force']
        self.seqlen_emb_type = kwargs['seqlen_emb']
        if kwargs['seqlen_emb'] == 'scratch':
            self.seqlen_emb = nn.Embedding(n_seqlens, gpt_nembd)
        elif 'sine' in kwargs['seqlen_emb']:
            pe = torch.zeros(n_seqlens, gpt_nembd)
            position = torch.arange(0, n_seqlens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, gpt_nembd, 2).float() * (-math.log(10000.0) / gpt_nembd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.seqlen_emb = nn.Parameter(pe)
            if 'frozen' in kwargs['seqlen_emb']:
                self.seqlen_emb.requires_grad = False


    def set_autoreg_head(self, gpt_nembd, kwargs):
        """ An autoregressive head models each index in product quantization autoregressively
        on the previous ones. """
        out_f, n_cb = n_e // kwargs['n_codebook'] + 1, kwargs['n_codebook']
        get_head = {'fc_wo_bias': lambda i: nn.Linear(in_features= i * (gpt_nembd // n_cb) + (n_cb - i) * out_f, out_features=out_f),
                    'mlp': lambda i: nn.Sequential(nn.Linear(in_features=i * (gpt_nembd // n_cb) + (n_cb - i) * out_f, out_features=out_f),
                                                   nn.ReLU(), nn.Linear(out_f, out_f))
                   }[kwargs['head_type']]
        self.autoreg_head = nn.ModuleDict({str(i): get_head(i) for i in range(1, n_cb)})

    def actions_to_embeddings(self, actions, factor):
        """
        Forward the embeddings.

        Args:
            - actions: k-hot encoding - [batch_size, n_actions] or [batch_size, seq_len, n_actions]
        Return:
            - emb: [batch_size, 1, emb_size] or  [batch_size, seq_len/factor, emb_size]
        """
        batch_size, *_ = actions.size()
        list_emb = []
        for i in range(batch_size):
            if len(actions.size()) == 2:
                actions = actions.unsqueeze(1)
            tdim = actions.size(1)
            emb = []
            for t in range(0, tdim, factor):
                idx = torch.where(actions[i][t])[0]
                if 'glove' in self.action_emb_type:
                    # inspired from https://arxiv.org/pdf/1912.06430.pdf
                    if idx.shape[0] == 0:
                        # use the <unk> embedding
                        idx_glove = torch.Tensor([self.action_emb.weight.shape[0] - 1]).type_as(idx)
                    else:
                        idx_glove = torch.cat([self.actionId2WordIds[idx_i.item()] for idx_i in idx]).type_as(idx)
                    emb_t = self.action_emb(idx_glove) # [?,50]
                    emb_t = self.action_emb_post(emb_t) # [?,256]
                    emb_t = F.max_pool1d(emb_t.permute(1,0).unsqueeze(1), emb_t.shape[0]).reshape(1, -1)
                elif 'scratch' in self.action_emb_type:
                    emb_t = self.action_emb(idx)
                    emb_t = emb_t.sum(0, keepdims=True)
                else:
                    raise NameError
                emb.append(emb_t)
            emb = torch.cat(emb)
            list_emb.append(emb)
        
        emb = torch.stack(list_emb)
        return emb

    def seqlens_to_embeddings(self, seqlens):
        """
        Forward the embeddings.

        Args:
            - seqlens: - [batch_size]
        Return:
            - emb: [batch_size, 1, emb_size]
        """
        batch_size = seqlens.shape[0]
        list_emb = []
        for i in range(batch_size):
            idx = seqlens[i].long()
            if 'sine' in self.seqlen_emb_type:
                emb = self.seqlen_emb[idx]
            elif 'scratch' in self.seqlen_emb_type:
                emb = self.seqlen_emb(idx)
            else:
                raise NameError
            list_emb.append(emb)
        emb = torch.stack(list_emb).unsqueeze(1)
        return emb


    # sample_transformer
    def gpt_sample_one_timestep(self, indices, actions_emb, seqlens_emb, sample_from_logits,
                                cut_to_seqlen=False):
        """ Sample from the logits. Possibly use auto-regression on pq factors.
        Indices are incremented by 1 to account for EOS symbol. """
        if cut_to_seqlen:
            if indices.shape[1] > self.seq_len:
                indices = indices[:, -self.seq_len:, ...]

        indices = indices + 1
        logits = self.gpt(indices, actions_emb, seqlens_emb)[:, -1, ...].unsqueeze(1) # bs, t, K, n_e_i
        if self.autoreg_pq:
            return self.sample_autoregressive_head(logits, sample_from_logits)
        else:
            return sample_from_logits(logits)

    def sample_autoregressive_head(self, logits, sample_from_logits):
        """ Each """
        time_dim = logits.shape[1]
        logits = rearrange(logits, 'bs t K n_e_i -> (bs t) K n_e_i').split(1, dim=-2)
        k = sample_from_logits(logits[0].unsqueeze(1)).unsqueeze(-1)
        idx_embed, indices = [], k
        for i in range(1, len(logits)):
            last_idx_embed = self.gpt.tok_emb(self.gpt.cat_indices(indices))[:, :, -1, :]
            idx_embed.append(rearrange(last_idx_embed, 'bs t K c -> (bs t) K c'))
            autoreg_inputs = torch.cat([*idx_embed[:i], *logits[i:]], dim=-1)
            klogits = self.autoreg_head[str(i)](autoreg_inputs)
            ik = rearrange(sample_from_logits(klogits.unsqueeze(1)), '(bs t) K tok -> bs t K tok', t=time_dim)
            indices = torch.cat([indices, ik], dim=2)
        return rearrange(indices, 'b t K one_dim -> (b t) one_dim K')


    @torch.no_grad()
    def sample(self, z, actions_emb, seqlens_emb, steps, temperature=1.0, sample=False, top_k=None, callback=lambda k: None,
            cut_to_seqlen=False):
        """ Sample indices for each time steps sequentially"""
        block_size = self.gpt.get_block_size()
        assert not self.gpt.training
        sample_from_logits = partial(_sample_from_logits, temperature=temperature, sample=sample, top_k=top_k)
        # Iterate steps times
        for k in range(steps):
            callback(k)
            assert z.size(1) <= block_size  # make sure model can see conditioning
            z_cond = z if z.size(1) <= block_size else z[:, -block_size:, ...]  # crop context if needed
            iz = self.gpt_sample_one_timestep(z_cond, actions_emb=actions_emb,
                    seqlens_emb=seqlens_emb, sample_from_logits=sample_from_logits,
                    cut_to_seqlen=cut_to_seqlen)
            # remove eos
            iz = iz - (1 if self.gen_eos else 0)
            # append to the sequence and continue
            z = torch.cat((z, iz), dim=1)
        return z

    @torch.no_grad()
    def sample_indices(self, zidx, actions_emb=None, seqlens_emb=None, temperature=None, top_k=None, cond_steps=0, ward=True,
            cond_end=False, sample=True, cut_to_seqlen=False, steps=None):
        """ Takes the [cond_steps] first elements of zidx, samples the rest. """
        assert zidx.shape[1] >= cond_steps, "Not enough conditioning data."
        if ward: self.gpt.eval()
        z_cond = zidx[:, :cond_steps, ...] if not cond_end else zidx[:, -cond_steps:, ...]
        steps = steps if steps is not None else zidx.shape[1] - z_cond.shape[1]
        index_sample = self.sample(z_cond, actions_emb=actions_emb, seqlens_emb=seqlens_emb, steps=steps,
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=sample, top_k=top_k, callback=lambda k: None,
                                    cut_to_seqlen=cut_to_seqlen)
        if ward: self.gpt.train()
        return index_sample

    @torch.no_grad()
    def sample_poses(self, zidx, x, valid, actions=None, seqlens=None, temperature=None,
            top_k=None, cond_steps=0, return_index_sample=False, return_zidx=False):
        """ Sample indices then forward propagate the decoder and body model to get poses. """
        actions_emb = self.actions_to_embeddings(actions, self.factor) if actions is not None else None
        seqlens_emb = self.seqlens_to_embeddings(seqlens) if seqlens is not None else None

        if zidx is None or cond_steps > 0 or zidx.shape[0] != actions_emb.shape[0]:
            _, zidx = self.vqvae.forward_latents(x, valid, return_indices=True)

        index_sample = self.sample_indices(zidx, actions_emb, seqlens_emb, temperature, top_k, cond_steps, ward=False)
        (rotmat, delta_trans), valid = self.forward_from_indices(index_sample, return_valid=True, eos=-1)
        trans = get_trans(delta_trans, valid=None)
        rotvec = roma.rotmat_to_rotvec(rotmat)
        out = [(rotvec, trans), valid]
        if return_index_sample:
            out.append(index_sample)
        if return_zidx:
            out.append(zidx)
        return out
