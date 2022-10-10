# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import numpy as np
from functools import partial
import smplx
import os
import yaml

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from utils.constants import SMPLX_DIR, mm
from utils.body_model import pose_to_vertices
from torch.utils.tensorboard import SummaryWriter
from utils.variable_length import repeat_last_valid
from utils.visu import visu_gt_rec
from argparse import ArgumentParser

from utils.body_model import SimplePreparator as Preparator

import roma

class Trainer():
    """ Basic trainer class.
    Used for both auto-encoders and gpt models.
    - Handles model loading / checkpointing.
    - Has a body model for visualisations.
    """
    def __init__(self, *, model, optimizer, device, args,
                 epoch, start_iter, type='smplx',
                 seq_len=16, loss_scaler=None):
        super(Trainer, self).__init__()
        self.args = args
        self.device = device
        self.optimizer = optimizer
        self.seq_len = seq_len
        self.bm = smplx.create(SMPLX_DIR, type, use_pca=False, batch_size=seq_len).to(device)
        self.loss_scaler = loss_scaler

        for param in self.bm.parameters():
            param.requires_grad = False

        self.faces = torch.as_tensor(np.array(c2c(self.bm.faces), dtype=np.int32), device=device)[None, :, :]
        self.type = type
        self.init_logdir()
        self.current_iter = start_iter
        self.current_epoch = epoch
        self.model = model
        self.model.seq_len = seq_len
        self.pose_to_vertices = partial(pose_to_vertices, pose_type=self.type,
                                        alpha=self.args.alpha_trans, bm=self.bm)
        self.classifier = None
        self.preparator = Preparator(seq_len=args.seq_len,
                                     mask_trans=args.mask_trans,
                                     pred_trans=args.pred_trans)

    def init_logdir(self):
        """ Create the log dir space and associated subdirs """
        log_dir_root = os.path.join(self.args.save_dir, self.args.name)
        os.makedirs(log_dir_root, exist_ok=True)
        self.args.log_dir = log_dir_root
        print(f"\n*** LOG_DIR = {self.args.log_dir} ***")

        self.args.ckpt_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        # tensorboard
        self.writer = SummaryWriter(self.args.log_dir)

        # save hparams
        with open(os.path.join(self.args.log_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, default_flow_style=False)

    def eval_pve(self, rotvec, rotmat_hat, trans_gt, trans_hat, valid, reduce_batch=True):
        """ Given predicted and GT body model parameters, compute [p]er [v]ertex [e]rror."""
        gt = torch.cat([rotvec.flatten(2), trans_gt], -1)
        pred = torch.cat([roma.rotmat_to_rotvec(rotmat_hat).flatten(2), trans_hat], -1)
        verts, verts_hat = [self.pose_to_vertices(x) for x in [gt, pred]]
        vsum = (lambda x: (x * valid).sum() / valid.sum()) if reduce_batch else lambda x: x
        pve = lambda x, y: (mm * torch.sqrt(((x - y) ** 2).sum(-1)).mean(-1))
        # PVE
        err = vsum(pve(verts_hat, verts))
        err_wo_trans = vsum(pve(verts_hat - trans_hat.unsqueeze(2), verts - trans_gt.unsqueeze(2)))
        return err, err_wo_trans, verts_hat, verts
    

    def sample_for_visu(self, bs, actions, valid, device):
        """ Sample smpl parameters then vertices"""
        # create a sample·
        (rotvec, trans), valid = self.model.sample_poses(bs, y=actions, valid=valid, device=device)
        verts = self.pose_to_vertices(torch.cat([rotvec.flatten(2), trans], -1))
        verts = repeat_last_valid(verts, valid)
        return {'sample': verts}

    def save_visu(self, verts_hat, verts, valid, samples, current_iter, save_to_tboard,
            save_to_disk=False, is_train=False, tag='generation_'):
        """ Save video of rendered smpl model to tensorboard. """
        visu_dir = os.path.join(self.args.log_dir, 'visu', f"{self.current_epoch:06d}")
        os.makedirs(visu_dir, exist_ok=True)

        nb_visu_saved = len(os.listdir(visu_dir)) if not self.args.debug else 0
        err = (mm * torch.sqrt(((verts_hat - verts) ** 2).sum(-1)))  # [batch_size, seq_len, vertices]
        samples = samples['sample'] if samples is not None else None

        i = 0
        offset = nb_visu_saved
        while nb_visu_saved < self.args.n_visu_to_save and i < verts.size(0):
            list_video = visu_gt_rec(err[i], valid[i], verts_hat[i], verts[i],
                    self.faces, self.device, visu_dir, nb_visu_saved,
                    sample=samples[i] if samples is not None else None,
                    save_to_disk=save_to_disk)
            if save_to_tboard:
                tboard_video_format = lambda x: np.transpose(
                    np.concatenate([np.expand_dims(a, axis=0) for a in x], axis=0), (0, 1, 4, 2, 3))
                _tag = ('train/' if is_train else '') + tag
                self.writer.add_video(_tag + str(offset + i), tboard_video_format(list_video),
                                      global_step=current_iter, fps=10, walltime=None)
            i += 1
            nb_visu_saved += 1
        return nb_visu_saved < self.args.n_visu_to_save

    def checkpoint(self, tag, extra_dict={}):
        """ Saving model and optimizer state """
        save_dict = {'epoch': self.current_epoch,
                     'iter': self.current_iter,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}

        if self.loss_scaler is not None:
            save_dict.update({'scaler': self.loss_scaler.state_dict()})

        if hasattr(self.model, 'quantizer'):
            save_dict.update({'balance_stats': self.model.quantizer.get_state()})

        save_dict.update(extra_dict)
        torch.save(save_dict, os.path.join(self.args.ckpt_dir, tag + ".pt"))

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mask_trans", type=int, default=0)
        parser.add_argument("--pred_trans", type=int, default=1)
        return parser
