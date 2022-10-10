# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from argparse import ArgumentParser
from itertools import product
import os
import sys
import time

import numpy as np
import roma
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_encode import Trainer
from models.transformer_vqvae import CausalVQVAE, OnlineVQVAE, TransformerVQVAE
from models.poseGPT import poseGPT
from dataset.mocap import MocapDataset, worker_init_fn
from evaluate import classification_evaluation
from utils.ae_utils import get_parameters, get_user, red
from utils.data import get_data_loaders
from utils.body_model import get_trans
from utils.checkpointing import get_last_checkpoint
from utils.constants import mm
from utils.param_count import print_parameters_count #benchmark, 
from utils.stats import AverageMeter, class_accuracy
from utils.variable_length import repeat_last_valid
from utils.visu import visu_sample_gt_rec
from utils.sampling import compute_fid
from utils.amp_helpers import NativeScalerWithGradNormCount as NativeScaler

class GTrainer(Trainer):
    """ Trainer for the generation step (training of the transformer that predicts latent variables) """
    def __init__(self, *, best_val=None, class_conditional=True, gen_eos=False, seqlen_conditional=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.best_val = 1e5 if best_val is None else best_val
        self.class_conditional = class_conditional
        self.seqlen_conditional = seqlen_conditional
        self.gen_eos = gen_eos

    def forward_one_batch(self, x, valid, actions, seqlens):
        """ Apply the model to a batch of data"""
        with torch.no_grad(): # auto encoder is already trained.
            self.model.vqvae.eval()
            _, zidx, zvalid = self.model.vqvae.forward_latents(x, valid, return_indices=True, return_mask=True)
            target, mzi = zidx, zidx

        factor = x.shape[1] // mzi.shape[1]
        actions_emb = self.model.actions_to_embeddings(actions, factor) if self.class_conditional else None
        seqlens_emb = self.model.seqlens_to_embeddings(seqlens) if self.seqlen_conditional else None
        logits = self.model.forward_gpt(mzi, actions_emb=actions_emb, seqlens_emb=seqlens_emb)
        assert logits.shape[:3] == target.shape, "Incoherent shapes, will likely be smartcasted wrong (dumbcasted)"

        if self.gen_eos:
            target_with_eos = target + 1
        else:
            # map (-1) to an arbitrary number (it will not be evaluated).
            target_with_eos = target
            target_with_eos[target == -1] = 0
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              target_with_eos.reshape(-1), reduction='none')

        nll = nll.reshape(logits.shape[:-1])
        if not self.gen_eos:
            nll = (nll * zvalid.unsqueeze(-1)).mean(-1).sum() / zvalid.sum()
        else:
            # TODO We should evaluate only one eos token, this gives eos too much weight.
            nll =  torch.mean(nll)
        return nll, target, logits, actions_emb, seqlens_emb, target

    def train_n_iters(self, data):

        print(f"TRAIN:")
        self.model.gpt.train()
        avg_nll, data_time, batch_time = [AverageMeter(k, ':6.3f') for k in ['Nll_latents', 'data_time', 'batch_time']]
        nll_meters = {'nll': avg_nll}

        end = time.time()
        print("> Training...")
        for x, valid, actions in tqdm(data):
            data_time.update(time.time() - end)
            x, valid, actions = x.to(self.device), valid.to(self.device), actions.to(self.device)
            x_noise, rotvec, rotmat, trans_gt, _, _ = self.preparator(x)

            x_gt = torch.cat([rotmat[..., :2].flatten(2), trans_gt], -1)
            seqlens = valid.sum(1)

            if self.args.use_amp:
                assert self.loss_scaler is not None
                nll, *_ = self.forward_one_batch(x_noise, valid, actions, seqlens)
                loss = nll
                self.optimizer.zero_grad()
                self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                                 update_grad=True)
            else:
                nll, *_ = self.forward_one_batch(x_noise, valid, actions, seqlens)
                loss = nll
                # optimization
                self.optimizer.zero_grad()
                loss.backward()

            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            avg_nll.update(nll)

            if self.current_iter % self.args.log_freq == 0 and self.current_iter > 0:
                for k, v in nll_meters.items():
                    self.writer.add_scalar(f"loss/{k}", v.avg, self.current_iter)
                    v.reset()

            self.current_iter += 1

        for k, v in nll_meters.items():
            print(f"    - {k}: {v.avg:.3f}")

    def eval(self, data, *, epoch, do_save_visu=False, save_to_tboard=False, do_compute_fid=False, do_classif_eval=False):
        avg_nll, avg_acc = AverageMeter('Nll_latents', ':6.3f'), AverageMeter('Acc', ':3.2f')
        meters = {'nll': avg_nll, 'acc': avg_acc}
        self.model.gpt.eval()
        x, valid, actions, zidx = None, None, None, None

        need_more_visu = do_save_visu
        with torch.no_grad():
            print(red("> Evaluation..."))
            for x, valid, actions in tqdm(data):
                self.model.vqvae.eval()
                x, valid, actions = x.to(self.device), valid.to(self.device), actions.to(self.device)
                x_noise, rotvec_gt, rotmat, trans_gt, _, _ = self.preparator(x)
                x_gt = torch.cat([rotmat[..., :2].flatten(2), trans_gt], -1)
                seqlens = valid.sum(1)

                nll, target, logits, actions_emb, seqlens_emb, zidx = self.forward_one_batch(x_noise,
                        valid, actions, seqlens)
                avg_nll.update(nll)

                target_with_eos = target + 1
                if self.class_conditional:
                    acc = class_accuracy(logits=logits, target_with_eos=target_with_eos)
                    avg_acc.update(acc)
                if do_save_visu and need_more_visu:
                    # Visu everything in a single video
                    need_more_visu = self.save_visu(rotvec_gt, trans_gt, zidx, actions_emb, seqlens_emb, valid,
                                                    self.current_iter, save_to_tboard=save_to_tboard)

        print(red(f"VAL:"))
        for k, v in meters.items():
            print(f"    - {k}: {v.avg:.3f}")
            self.writer.add_scalar('val/' + k, v.avg, self.current_iter)

        if do_compute_fid:
            compute_fid(self.model, data, self.preparator, self.device, self.args.classif_ckpt,
                    writer=self.writer, current_iter=self.current_iter, debug=self.args.debug)

        if do_classif_eval:
            mAP_reals = classification_evaluation(self.model, data, self.args.log_dir,
                                                  epoch, self.args, self.args.classif_ckpt,
                                                  while_training=True,
                                                  preparator=self.preparator,
                                                  data_type=self.args.dataset_type)
            self.writer.add_scalar('val/mAP_real' , mAP_reals, self.current_iter)

        self.model.gpt.train()
        return avg_nll.avg


    def fit(self, data_train, data_val):
        """
        Train and evaluate a model using training and validation data
        """
        while self.current_epoch <= self.args.max_epochs:
            epoch = self.current_epoch
            sys.stdout.flush()

            print(f"\nEPOCH={epoch:03d}/{self.args.max_epochs} - ITER={self.current_iter}")
            self.train_n_iters(data_train)
            if epoch % self.args.val_freq == 0:
                # Validate the model
                do_compute_fid = self.args.fid_freq > 0 and epoch % self.args.fid_freq == 0 and epoch > 0
                do_classif_eval = self.args.class_freq > 0 and epoch % self.args.class_freq == 0 and epoch > 0
                val = self.eval(data_val, epoch=epoch,
                                do_save_visu=(epoch % self.args.visu_freq == 0),
                                save_to_tboard=self.args.visu_to_tboard,
                                do_compute_fid=do_compute_fid,
                                do_classif_eval = do_classif_eval)

                # Save ckpt
                if val < self.best_val:
                    self.checkpoint(tag='best_val', extra_dict={'pve': val})
                    self.best_val = val


            if epoch % self.args.ckpt_freq == 0 and epoch > 1:
                self.checkpoint(tag='ckpt_' + str(epoch), extra_dict={'best_val': self.best_val})

            if epoch % self.args.restart_ckpt_freq == 0 and epoch > 1:
                self.checkpoint(tag='ckpt_restart', extra_dict={'best_val': self.best_val})

            self.current_epoch += 1
        return None

    def verts_from_indices(self, index):
        """ Given token indices, forward the model and body model, convert to vertices. """
        (rotmat, delta_trans), valid = self.model.forward_from_indices(index, return_valid=True, eos=-1)
        trans = get_trans(delta_trans, valid=None)
        rotvec = roma.rotmat_to_rotvec(rotmat)
        verts = self.pose_to_vertices(torch.cat([rotvec.flatten(2), trans], -1))
        return repeat_last_valid(verts, valid)

    def verts_sample(self, zidx, actions_emb=None, seqlens_emb=None, temperature=None, top_k=None, cond_steps=0, return_indices=False):
        """ Sample indices, forward propagate to vertices.
        Args:
            - zidx: [32, 32, 32]
            - actions_emb: [32, 1, 512]
            - seqlens_emb: [32, 1, 512]
        """
        index_sample = self.model.sample_indices(zidx, actions_emb, seqlens_emb, temperature, top_k, cond_steps)
        verts_samples = self.verts_from_indices(index_sample)
        if return_indices:
            return verts_samples, index_sample
        return verts_samples

    def sample_for_visu(self, zidx, actions_emb=None, seqlens_emb=None, temperature=None, top_k=None):
        """ Sample vertices with and without conditioning on half the sequence of tokens."""
        # create a sample and a "half"" sample.
        full = self.verts_sample(zidx, actions_emb=actions_emb,
                                 seqlens_emb=seqlens_emb,
                                 temperature=temperature, top_k=top_k,
                                 cond_steps=0)
        half = self.verts_sample(zidx, actions_emb=actions_emb,
                                 seqlens_emb=seqlens_emb,
                                 temperature=temperature, top_k=top_k,
                                 cond_steps=zidx.shape[1] // 2)
        return {'sample': full, 'half_sample': half}

    def save_visu(self, rotvec, trans_gt, zidx, actions_emb, seqlens_emb, valid, current_iter, save_to_tboard):
        """
        Visu sample, half_sample, upper-bound from zid, real sample
        """
        visu_dir = os.path.join(self.args.log_dir, 'visu', f"{self.current_epoch:06d}")
        os.makedirs(visu_dir, exist_ok=True)

        verts = self.pose_to_vertices(torch.cat([rotvec.flatten(2), trans_gt], -1)) # GT
        vsamples = self.sample_for_visu(zidx, actions_emb=actions_emb, seqlens_emb=seqlens_emb) # Samples
        verts_upper_bound = self.verts_from_indices(zidx) # GT upper bound
        verts_sample, verts_half_sample = vsamples['sample'], vsamples['half_sample']

        nb_visu_saved = len(os.listdir(visu_dir))
        err = (mm * torch.sqrt(((verts_upper_bound - verts) ** 2).sum(-1)))  # [batch_size, seq_len, vertices]
        i = 0
        offset = nb_visu_saved
        while nb_visu_saved < self.args.n_visu_to_save and i < verts.size(0):
            list_video = visu_sample_gt_rec(verts_sample[i], verts_half_sample[i], verts_upper_bound[i],
                                            err[i], valid[i], verts[i], self.faces, self.device, visu_dir, nb_visu_saved)
            if save_to_tboard:
                tboard_video_format = lambda x: np.transpose(
                    np.concatenate([np.expand_dims(a, axis=0) for a in x], axis=0), (0, 1, 4, 2, 3))
                # vid_tensor: (N, T, C, H, W)(N,T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float
                self.writer.add_video('generation_' + str(offset + i), tboard_video_format(list_video),
                                      global_step=current_iter, fps=10, walltime=None)

            i += 1
            nb_visu_saved += 1
        return nb_visu_saved < self.args.n_visu_to_save

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = super(GTrainer, GTrainer).add_trainer_specific_args(parent_parser)
        parser.add_argument("--alpha_root",
                            type=float)  # Not giving them default values to make sure we don't forget. 
        parser.add_argument("--alpha_body",
                            type=float)  # Not giving them default values to make sure we don't forget.
        parser.add_argument("--alpha_trans", type=float, default=1)
        return parser

def get_parsers_and_models(args):
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=600)
    #parser.add_argument("--data_device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dummy_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--overfit", type=int, default=0)

    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--class_freq", type=int, default=100)
    parser.add_argument("--visu_freq", type=int, default=200)
    parser.add_argument("--visu_to_tboard", type=int, default=int(get_user() == 'tlucas'), choices=[0,1])
    parser.add_argument("--ckpt_freq", type=int, default=50)
    parser.add_argument("--fid_freq", type=int, default=40)
    parser.add_argument("--restart_ckpt_freq", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--n_visu_to_save", type=int, default=0)
    parser.add_argument("--train_data_dir", type=str, default='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--val_data_dir", type=str, default='data/smplx/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--n_train", type=int, default=1000000)
    parser.add_argument("--n_iter_val", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default='logs')
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-b_train", type=int, default=32)
    parser.add_argument("--val_batch_size", "-b_val", type=int, default=16)
    parser.add_argument("--vq_model", type=str, default='OnlineVQVAE')
    parser.add_argument('--vq_ckpt', type=str, default=None)
    parser.add_argument("--model", type=str, default='generator.GPT_Transformer')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--detailed_count", type=int, default=0)
    parser.add_argument("--class_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seqlen_conditional", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gen_eos", type=int, default=0, choices=[0,1])
    parser.add_argument("--eos_force", type=int, default=1, choices=[0,1])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    parser.add_argument("--data_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_start", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_classif", type=int, default=0, choices=[0, 1])
    parser.add_argument("--concat_emb", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_fid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--head_type", type=str, default='fc_wo_bias', choices=['fc_wo_bias', 'mlp'])
    parser.add_argument("--pos_emb", type=str, default='scratch', choices=['scratch', 'sine_frozen', 'sine_ft', 'scratch_frozen'])
    parser.add_argument("--seqlen_emb", type=str, default='scratch', choices=['scratch', 'sine_frozen', 'sine_ft', 'scratch_frozen'])
    parser.add_argument("--action_emb", type=str, default='scratch', choices=['scratch', 'glove_frozen', 'glove_ft', 'scratch_frozen'])

    parser.add_argument("--classif_ckpt", type=str, default=None)
    parser.add_argument("--classif_ckpt_babel", type=str, default='/scratch/1/user/tlucas/pose_generation/logs/publish_transformer_classif_lrx2/classification_TR/checkpoints/ckpt_fid.pt')
    parser.add_argument("--use_amp", type=int, default=0, choices=[0, 1])

    script_args, _ = parser.parse_known_args(args)
    assert script_args.vq_model in ['CausalVQVAE', 'OnlineVQVAE', 'OfflineVQVAE'], "Invalid VQVAE model"
    VQModel = eval(script_args.vq_model)
    Model = eval(script_args.model)
    parser = GTrainer.add_trainer_specific_args(parser)
    parser = VQModel.add_model_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    return parser, VQModel, Model

def get_data(args, user):
    if args.debug:
        args.val_freq = 1
        args.visu_freq = 1
        args.fid_freq = 1
        args.class_freq = 1
        args.n_iters_per_epoch = 20
        args.train_batch_size = 23  # Hard to confuse with an other dimension
        args.train_data_dir = 'data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16_nMax1000'
        args.val_data_dir = 'data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16_nMax1000'
        args.n_iter_val = 2

        if not user == 'tlucas':
            args.visu_freq = 100
            args.n_iters_per_epoch = 2
            args.vq_ckpt = 'logs/vqvae/debug/tb/checkpoints/best_val.pt'
            args.n_codebook = 1
            args.n_e = 1
            args.n_layers = 2
            args.depth = 2
            args.pool_kernel = 1

    # Data
    print(f"\nLoading data...")
    loader_train, loader_val = get_data_loaders(args)

    print(f"Data - N_train={len(loader_train.dataset.pose)} - N_val={len(loader_train.dataset.pose)}")
    return loader_train, loader_val


def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user = get_user()
    parser, VQModel, Model = get_parsers_and_models(args)

    args = parser.parse_args(args)
    #args.factor = np.prod(args.pool_kernel) # temporal downsampling
    args.factor = 2 if isinstance(args.n_layers, int) else 2 ** len(args.n_layers)

    loader_train, loader_val = get_data(args, user)

    known_dirs_to_classifier = {'babel': args.classif_ckpt_babel}
    matching = [k for k in known_dirs_to_classifier.keys() if k in loader_train.dataset.data_dir]
    args.dataset_type = matching[0] if len(matching) else 'unknown'
    if args.classif_ckpt is None:
        assert len(matching) == 1, "Unknow data dir, provide classif_ckpt manually"
        args.classif_ckpt = known_dirs_to_classifier[args.dataset_type]


    print(f"\nBuilding the quantization model...")
    print(args)

    in_dim = ((loader_train.dataset.pose[0].size(1) // 3) - 1) * 6 + 3  # jts in 6D repr, trans in 3d coord
    vq_model = VQModel(in_dim=in_dim, **vars(args)).to(device)

    assert args.vq_ckpt is not None, "You should use a pretrained VQ-VAE"
    vq_ckpt = torch.load(args.vq_ckpt)
    weights = vq_ckpt['model_state_dict']
    weights = {k.replace('log_sigmas.verts', 'log_sigmas.vert'): v for (k, v) in weights.items()}
    missing, unexpected = vq_model.load_state_dict(weights, strict=False)
    assert len(unexpected) == 0, "Unexpected keys"
    assert all(['log_sigmas' in m for m in missing]), "Missing keys: " + ','.join([m for m in missing if 'log_sigmas' not in m])

    if 'balance_stats' in vq_ckpt.keys():
        bins = vq_ckpt['balance_stats']
        vq_model.quantizer.load_state(bins)


    model = Model(**vars(args), vqvae=vq_model).to(device)

    print("VQ model parameter count: ")
    print_parameters_count(model.vqvae, detailed=args.detailed_count, tag='VQ - ')
    print("Transformer model parameter count: ")
    print_parameters_count(model.gpt, detailed=args.detailed_count, tag='GPT - ')

    print(f"Number of parameters: {get_parameters(model):,}")
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        ckpt_path = args.pretrained_ckpt
    else:
        checkpoint, ckpt_path = get_last_checkpoint(args.save_dir, args.name)

    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        assert len(unexpected) == 0, "Unexpected keys"
        assert all(['log_sigmas' in m for m in missing]), "Missing keys: " + ','.join([m for m in missing if 'log_sigmas' not in m])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_scaler = NativeScaler() if args.use_amp else None

    if checkpoint is not None:
        if not (args.eval_classif or args.eval_fid):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch, saved_iter = [checkpoint[k] for k in ['epoch', 'iter']]
        bv, bc = [checkpoint[k] if k in checkpoint else None for k in ['best_val', 'best_class']]
        print(f"Ckpt succesfully loaded from: {ckpt_path}")
        if 'scaler' in checkpoint:
            assert loss_scaler is not None, "I have found weights for the loss_scaler, but don't have it."
            loss_scaler.load_state_dict(checkpoint['scaler'])
    else:
        epoch, saved_iter = 1, 0
        bv, bc = None, None

    # Trainer
    print(f"\nSetting up the trainer...")
    trainer = GTrainer(model=model, optimizer=optimizer, device=device,
                       args=args, epoch=epoch, start_iter=saved_iter,
                       best_val=bv, type=loader_train.dataset.type,
                       class_conditional=args.class_conditional,
                       seqlen_conditional=args.seqlen_conditional,
                       gen_eos=args.gen_eos,
                       seq_len=args.seq_len, 
                       loss_scaler=loss_scaler)

    if args.eval_classif or args.eval_fid:
        data_loader = DataLoader(MocapDataset(data_dir=args.train_data_dir,
                                              seq_len=args.seq_len, training=False,
                                              n_iter=None,
                                              n=-1,
                                              data_augment=0,
                                              dummy=0),
                                 batch_size=32, num_workers=1,
                                 prefetch_factor=2, shuffle=False,
                                 worker_init_fn=worker_init_fn,
                                 pin_memory=False, drop_last=True)
        if args.eval_classif:
            classification_evaluation(model, data_loader,
                                      trainer.args.log_dir,
                                      epoch, args,
                                      args.classif_ckpt,
                                      preparator=self.preparator,
                                      while_training=False)

        elif args.eval_fid:
            raise NotADirectoryError("Eval only needs to be implemented")
    else:
        trainer.fit(loader_train, loader_val)


if __name__ == "__main__":
    main()
