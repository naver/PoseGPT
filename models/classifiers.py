# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np
import torch
from torch import nn
import smplx
from utils.constants import SMPLX_DIR
from threed.skeleton import get_smplx_pose, get_smplh_pose, get_smpl_pose
from models.transformer_vqvae import Stack

class Autoreg_classifier(nn.Module):
    """ An autoregressive model, working directly on smpl paramaters. """
    def __init__(self, input_dim, mlp_dim=120,
                 gru_dim=256, use_last=False, nb_classes=60):
        super().__init__()
        self.gru = nn.GRU(input_dim, gru_dim, 1,
                          batch_first=True, dropout=0,
                          bidirectional=False)
        self.fc1 = nn.Linear(gru_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, nb_classes)
        self.use_last = use_last

    def forward_backbone(self, x, valid):
        gru_output, _ = self.gru(x.flatten(2))
        out = slice_last_valid(valid, gru_output) if self.use_last else gru_output
        hid = self.fc1(out)
        hid = torch.tanh(hid)
        return hid

    def forward(self, x, valid):
        hid = self.forward_backbone(x, valid)
        logits = self.fc2(hid)
        if not self.use_last:
            logits = (logits * valid.unsqueeze(-1)).sum(1) / valid.sum(-1).unsqueeze(-1)
        return logits

    def forward_fid(self, x, valid):
        """ Return penultimate activations.
        Average then when not using last activations to get constant dimensions"""
        hid = self.forward_backbone(x, valid)
        logits = self.fc2(hid)

        if not self.use_last:
            hid = (hid * valid.unsqueeze(-1)).sum(1) / valid.sum(-1).unsqueeze(-1)
            logits = (logits * valid.unsqueeze(-1)).sum(1) / valid.sum(-1).unsqueeze(-1)
        return logits, hid


class TransformerDiscriminator(nn.Module):
    """ Classifier that uses a body model to define inputs. """
    def __init__(self, device, in_dim, block_size=2048, n_layer=4, n_head=4, n_embd=256, dropout=0.1, pos_type='sine_frozen', down=2, output_size=60):
        super(TransformerDiscriminator, self).__init__()
        self.device = device

        self.emb = nn.Linear(in_dim, n_embd)
        self.stack = Stack(block_size=block_size, n_layer=n_layer, n_head=n_head,
                           n_embd=n_embd, pos_type=pos_type, down=down,
                           dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(n_embd, n_embd),
                                 nn.ReLU(), nn.Linear(n_embd, output_size))

    def forward_backbone(self, x, valid):
        # Pad x and valid by 1, to use as classification token.
        x, valid = [torch.cat([torch.ones_like(t[:, 0,...][:, None, ...]), t], dim=1) for t in [x, valid]]
        out = self.stack(self.emb(x), mask=valid)[0]
        return out

    def forward(self, x, valid):
        # dim (num_samples, output_size)
        hid = self.forward_backbone(x, valid)
        logits = self.proj(hid[:, 0, :])
        return logits

    def forward_fid(self, x, valid):
        # dim (num_samples, output_size)
        hid = self.forward_backbone(x, valid)
        logits = self.proj(hid[:, 0, :])
        return logits, hid[:, 0, :]

class bm_classifier(nn.Module):
    """ Wrapper around classifier that changes inputs from smpl to joints."""
    def __init__(self, classifier, time_dim, proportion, batch_size, device,
                 joints_parallel=True, pose_type='smplx', cut_joints=None, center_at_pelvis=True):
        super(bm_classifier, self).__init__()
        self.classifier = classifier
        self.method = classifier.method
        self.pose_type = pose_type
        self.proportion = proportion
        self.joints_parallel = joints_parallel
        self.bs = batch_size
        self.cut_joints = cut_joints
        self.center_at_pelvis = center_at_pelvis
        if joints_parallel:
            self.tdim = int(time_dim * proportion)
            self.bm = smplx.create(SMPLX_DIR, self.pose_type, use_pca=False,
                                   batch_size=batch_size * self.tdim).to(device)
        else:
            self.bm = smplx.create(SMPLX_DIR, self.pose_type, use_pca=False,
                                   batch_size=self.time_dim).to(device)
        for param in self.bm.parameters():
            param.requires_grad = False

    def forward(self, x, valid):
        if not self.joints_parallel:
            raise NotImplementedError
        else:
            period = int(1. / self.proportion)
            offset = np.random.randint(0, period)
            _x, _valid = [t[:, offset::period, ...][:, :self.tdim] for t in [x, valid]]
            gp = {'get_smplx_pose': get_smplx_pose, 'get_smplh_pose': get_smplh_pose, 'get_smpl_pose': get_smpl_pose,
                  }[f"get_{self.pose_type}_pose"]
            mask_pad = None
            if int(_x.shape[0]) != self.bs:
                mask_pad = self.bs - _x.shape[0]
                _x = torch.cat((_x, torch.zeros((self.bs - _x.shape[0], *_x.shape[1:])).to(_x.device)), dim=0)
            pose = gp(_x.reshape(-1, x.shape[-1]))
            pose['betas'] = torch.zeros(_x.shape[0]* _x.shape[1], 10).type_as(_valid).float()
            motion_sequence = self.bm(**pose).joints.detach()

        bs, num_frames, njoints, nfeats = self.bs, self.tdim, motion_sequence.shape[-2], motion_sequence.shape[-1]
        if self.pose_type is not None and self.pose_type in ['smpl', 'smplx']:
            if self.cut_joints is not None:
                motion_sequence = motion_sequence[:, np.arange(self.cut_joints)] # keep only the [cut_joints] first joints
            if self.center_at_pelvis:
                motion_sequence = motion_sequence - motion_sequence[:, [0]] # center around pelvis
            njoints = motion_sequence.shape[1]

        motion_sequence = motion_sequence.reshape(bs, num_frames, njoints*nfeats)
        if mask_pad:
            motion_sequence = motion_sequence[:(self.bs - mask_pad), ...]
        return self.classifier(motion_sequence, valid=_valid)

