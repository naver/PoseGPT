# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from argparse import ArgumentParser
from functools import partial
import sys
import time
import warnings

import numpy as np
import roma
import smplx
import torch
from tqdm import tqdm

from models.transformer_vqvae import CausalVQVAE, OnlineVQVAE, TransformerVQVAE, OfflineVQVAE
from trainer import Trainer
from utils.ae_losses import gaussian_nll, laplacian_nll
from utils.ae_utils import get_parameters, get_user, red
from utils.data import get_data_loaders
from utils.body_model import get_trans, pose_to_vertices
from utils.checkpointing import get_last_checkpoint
from utils.constants import SMPLX_DIR, MB
from utils.log_helpers import add_histogram
from utils.param_count import print_parameters_count
from utils.stats import AverageMeter
from utils.utils import (count_dim, subsamble_random_offset,
    valid_reduce as _valid_reduce)
from utils.amp_helpers import NativeScalerWithGradNormCount as NativeScaler

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class QTrainer(Trainer):
    """ Trainer specialized for the auto-encoder based quantization step. """

    def __init__(self, *, best_val=None, best_class=None, **kwargs):
        super().__init__(**kwargs)

        self.best_val = 1e5 if best_val is None else best_val
        self.best_class = -1e5 if best_class is None else best_class

        if hasattr(self.args, 'tprop_vert') and self.args.tprop_vert != 1.:
            """ Apply loss on vertices, for certain time steps only"""
            self.tdim = int(self.seq_len * self.args.tprop_vert)
            nb_poses = self.tdim * int(self.args.train_batch_size * self.args.prop_vert)
            self.bm_light = smplx.create(SMPLX_DIR, self.type, use_pca=False, batch_size=nb_poses).to(self.device)
            self.pose_to_vertices_light = partial(pose_to_vertices, pose_type=self.type,
                                                  alpha=self.args.alpha_trans, bm=self.bm_light,
                                                  parallel=True)
        elif self.args.alpha_vert > 0.:
            self.tdim = self.seq_len
            print("Warning: this will be slow. If you are sure, discard & proceed.")
            sys.exit(0)

    def forward_one_batch(self, x, actions, valid, loss_type, trans_gt, rotmat, rotvec, training=True):
        # Forward model
        (rotmat_hat, trans_delta_hat), loss_z, indices = self.model(
            x=x, y=actions, valid=valid)
        trans_hat = get_trans(trans_delta_hat, valid)

        verts, verts_hat, verts_valid = None, None, None
        if training:
            # Convert smpl sequence to sequence of vertices
            verts, verts_hat, verts_valid = self.params_to_vertices(rotmat_hat, rotvec, trans_gt, trans_hat, valid)

        # Define masked reductions
        def valid_reduce(x, mask=None, reduction='sum'):
            """ Accounts for 0 padding in shorter sequences """
            if len(x.shape) == 1 and x.shape[0] == 1: # Just a scalar
                return x.sum() if reduction == 'sum' else x.mean()
            mask = (valid if valid.shape[1] == x.shape[1] else
                    verts_valid) if mask is None else mask
            return _valid_reduce(x, mask, reduction)

        valid_sum = partial(valid_reduce, reduction='sum')
        valid_mean = partial(valid_reduce, reduction='mean')

        # Predictions, targets and ground truths
        pred = {'trans': trans_hat, 'vert': verts_hat, 'body': rotmat_hat[:, :, 1:, ...],
                'root': rotmat_hat[:, :, 0, ...].unsqueeze(2)}
        gt = {'trans': trans_gt, 'vert': verts, 'body': rotmat[:, :, 1:, ...],
              'root': rotmat[:, :, 0, ...].unsqueeze(2)}

        # Variance can be learned; not used by default
        log_sigmas = self.model.log_sigmas if loss_type in ['gaussian', 'laplacian'] else \
            {k: torch.zeros((1), requires_grad=False).to(x.device) for k in pred.keys()}

        nll_loss = gaussian_nll if loss_type in ['gaussian', 'l2'] else laplacian_nll
        nll_verts_loss = laplacian_nll if not self.args.l2_verts else gaussian_nll
        nll = {k: nll_loss(pred[k], gt[k], log_sigmas[k]) for k in ['trans', 'root', 'body']}
        nll.update({'vert': nll_verts_loss(pred['vert'], gt['vert'], log_sigmas['vert'])})

        # Energy and norm seperated for logging;
        energy_values, norm_values, nll_values = [{k: nll[k][n] for k in nll.keys()} for n in ['energy', 'norm', 'nll']]

        # Elbo computations (for logging, not optimized by sgd)
        elbo_params, elbo_verts, valid_kl = self.compute_elbos(nll_values, valid_sum, loss_z)

        # Gather losses and multiply by the right coefficients
        losses = ['root', 'body', 'trans'] + (['vert'] if pred['vert'] is not None else [])
        total_loss = sum([getattr(self.args, 'alpha_' + k) * valid_mean(nll_values[k]).mean(0) for k in losses])
        total_loss += self.args.alpha_codebook * loss_z['quant_loss']

        # Putting usefull statistics together (for tensorboard)
        statistics = {'elbo/valid_kl': valid_kl, 'elbo/params': elbo_params, 'elbo/verts': elbo_verts,
                      'total_loss': total_loss, 'quant_loss': loss_z['quant_loss'] if 'quant_loss' in loss_z else 0}
        for tag, vals in zip(['', '_energy', '_norm'], [nll_values, energy_values, norm_values]):
            statistics.update({'nll/' + k + tag: valid_mean(vals[k]).mean(0) for k in nll_values.keys()})

        # Return pose predictions (smpl and vertices), centroid indices and mask)
        outputs = {'rotmat_hat': rotmat_hat, 'trans_hat': trans_hat, 'indices': indices, 'valid': valid}
        return total_loss, statistics, outputs

    def compute_elbos(self, nll_values, valid_sum, loss_z):
        """ Elbos are usefull for logging (put reconstruction and KL together) in principle."""
        params_dim = sum([count_dim(nll_values[k]) for k in ['root', 'body', 'trans']])
        valid_kl = valid_sum(loss_z['kl'], loss_z['kl_valid'])
        elbo_params = - (torch.stack([valid_sum(nll_values[k])
                         for k in ['root', 'body', 'trans']]).sum(0) + valid_kl).mean(0) / params_dim
        elbo_verts = torch.zeros((1), requires_grad=False).to(self.device)
        if 'vert' in nll_values and len(nll_values['vert'].shape) > 1:
            elbo_verts = - (valid_sum(nll_values['vert']) + valid_kl).mean(0) / count_dim(nll_values['vert'])
        return elbo_params, elbo_verts, valid_kl

    def params_to_vertices(self, rotmat_hat, rotvec, trans_gt, trans_hat, valid, use_fast_smpl=False):
        """ Maps smpl parameters to vertices for some time steps """
        tag = '' if not use_fast_smpl else 'fast_'
        prop_vert, freq_vert = [getattr(self.args, tag + k) for k in  ['prop_vert', 'freq_vert']]
        verts, verts_hat, verts_valid = None, None, None
        compute_vert = (self.current_iter % freq_vert == 0)
        if self.args.alpha_vert > 0 and compute_vert:
            rotvec_hat = roma.rotmat_to_rotvec(rotmat_hat)
            bs = int(self.args.train_batch_size * prop_vert)
            period = int(1 / self.args.tprop_vert)
            # Select evenly spaced time steps, with random offset at start, to compute faster.
            _rotvec, _trans, _rotvec_hat, _trans_hat, verts_valid = subsamble_random_offset(bs, period, self.tdim,
                    [rotvec, trans_gt, rotvec_hat, trans_hat, valid])
            ptv = self.pose_to_vertices if self.args.tprop_vert == 1.0 else self.pose_to_vertices_light
            verts, verts_hat = [ptv(torch.cat([r.flatten(2), t], -1))
                                for r, t in zip([_rotvec, _rotvec_hat], [_trans, _trans_hat])]
        return verts, verts_hat, verts_valid

    def train_n_iters(self, data, loss_type):
        """ Do a pass on the dataset; sometimes log statistics"""
        self.model.train()

        data_time, batch_time, max_mem = [AverageMeter(k, ':6.3f') for k in ['data_time', 'batch_time', 'max_mem']]
        average_meters = {'data_time': data_time, 'batch_time': batch_time}

        end = time.time()
        print(red("> Training auto-encoder..."))
        for x, valid, actions in tqdm(data):
            data_time.update(time.time() - end)

            # Input preparation
            x, valid = x.to(self.device), valid.to(self.device)
            #x, rotvec, rotmat, trans_gt, _ = prepare_input(x)
            x_noise, rotvec, rotmat, trans_gt, _, _ = self.preparator(x)


            # TODO refactor with a context manager to avoid code duplication.
            if self.args.use_amp:
                assert self.loss_scaler is not None, "Need a loss scaler for AMP."
                with torch.cuda.amp.autocast():
                    total_loss, statistics, outputs = self.forward_one_batch(x_noise, actions, valid,
                            loss_type, trans_gt, rotmat, rotvec)
                self.optimizer.zero_grad()
                self.loss_scaler(total_loss, self.optimizer, parameters=self.model.parameters(),
                                 update_grad=True)

            else:
                total_loss, statistics, outputs = self.forward_one_batch(x_noise, actions,
                        valid, loss_type, trans_gt, rotmat, rotvec)

                # optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Track time and memory
            batch_time.update(time.time() - end)
            max_mem.update(torch.cuda.max_memory_allocated() / (MB))
            end = time.time()

            # Update metrics
            if len(average_meters) == 2:
                average_meters.update({k: AverageMeter(k, ':6.3f') for k in statistics.keys()})
            for k in statistics.keys():
                average_meters[k].update(statistics[k].mean())
            self.log_train_statistics(average_meters)
            self.current_iter += 1

        for k, v in average_meters.items():
            if 'nll' in k or 'total' in k:
                print(f"    - {k}: {v.avg:.3f}")
        self.log_compute_efficiency(batch_time, data_time, max_mem)

    def visu_rec(self, rotvec, trans_gt, outputs, valid, epoch, is_train=True):
        gt = torch.cat([rotvec.flatten(2), trans_gt], -1)
        pred = torch.cat([roma.rotmat_to_rotvec(outputs['rotmat_hat']).flatten(2), outputs['trans_hat']], -1)
        verts, verts_hat = [self.pose_to_vertices(x) for x in [gt, pred]]
        samples = None
        self.save_visu(verts_hat, verts, valid, samples,
                self.current_iter, self.args.visu_to_tboard, is_train=is_train,
                tag='auto_encoding')

    def log_train_statistics(self, average_meters):
        """ Log statistics to tensorboard and console, reset the average_meters. """
        if not (self.current_iter % (self.args.log_freq - 1) == 0 and self.current_iter > 0):
            return
        for k, v in average_meters.items():
            self.writer.add_scalar(f"{k}", v.avg, self.current_iter)
            v.reset()
        for k, v in self.model.log_sigmas.items():
            self.writer.add_scalar(f"log_sigmas/{k}", v.data.detach(), self.current_iter)

        # Tracking centroid usage with histograms and a score
        centroid_balance_scores = []
        if hasattr(self.model, 'quantizer'):
            for k in self.model.quantizer.embeddings.keys():
                hist = self.model.quantizer.get_hist(int(k))
                if hist is not None:
                    hist = hist.cpu().numpy()
                    add_histogram(writer=self.writer, tag='train_stats/z_histograms_' + k,
                                  hist=hist, global_step=self.current_iter)
                    centroid_balance_scores.append(1 - np.abs((1 - hist*hist.shape[-1])).mean())
            self.writer.add_scalar(f"centroid_balance_score", np.mean(centroid_balance_scores), self.current_iter)

    def log_compute_efficiency(self, batch_time, data_time, max_mem):
        """ Measuring computation efficiency """
        self.writer.add_scalar(f"gpu_load/batch_time", batch_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/it_per_sec", 1. / (batch_time.avg +
                               (1e-4 if self.args.debug else 1e-4)), self.current_iter)
        self.writer.add_scalar(f"gpu_load/data_time", data_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/max_mem", max_mem.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/max_mem_ratio", max_mem.avg / (torch.cuda.memory_reserved() / MB), self.current_iter)
        print(f"    - batch_time: {batch_time.avg:.3f}")
        print(f"    - data_time: {data_time.avg:.3f}")
    
    def eval(self, data, *, loss_type, epoch, save_to_tboard):
        """ Run the model on validation data; no optimization"""
        pve, pve_wo_trans, pve_diff = [AverageMeter(k, ':6.3f') for k in ['pve', 'pve_wo_trans', 'pve_diff']]
        average_meters = {'pve': pve, 'pve_wo_trans': pve_wo_trans, 'pve_diff': pve_diff}

        self.model.eval()
        nb_visu_saved, need_more_visu = 0, True
        with torch.no_grad():
            print(red("> Evaluating auto-encoder..."))
            for x, valid, actions in tqdm(data):
                x, valid = x.to(self.device), valid.to(self.device)
                x_noise, rotvec, rotmat, trans_gt, _, _ = self.preparator(x)
                _, statistics, outputs = self.forward_one_batch(x_noise, actions, valid, loss_type,
                                                                trans_gt, rotmat, rotvec, training=False)
                err, err_wo_trans, verts_hat, verts = self.eval_pve(
                    rotvec, outputs['rotmat_hat'], trans_gt, outputs['trans_hat'], valid)

                # Logging
                if len(average_meters) == 3:
                    average_meters.update({k: AverageMeter(k, ':6.3f') for k in statistics.keys()})
                for k in statistics.keys():
                    average_meters[k].update(statistics[k].mean())
                for k, v in zip([pve, pve_wo_trans, pve_diff], [err, err_wo_trans, err - err_wo_trans]):
                    k.update(v)

                # Save visu
                do_visu = self.args.n_visu_to_save > 0 and nb_visu_saved < self.args.n_visu_to_save and epoch % self.args.visu_freq == 0
                if do_visu and need_more_visu:
                    samples = None
                    self.save_visu(verts_hat, verts, valid, samples, self.current_iter, save_to_tboard)
                    need_more_visu = False

            print(red(f"VAL:"))
            for k, v in average_meters.items():
                print(f"    - {k}: {v.avg:.3f}")
                self.writer.add_scalar(('val/' + k) if 'pve' not in k else ('pves/' + k), v.avg, self.current_iter)
                if k == 'pve':
                    self.writer.add_scalar('pve', v.avg, self.current_iter)
        return pve.avg

    def fit(self, data_train, data_val, *, loss='l2'):
        """
        Train and evaluate a model using training and validation data
        """
        while self.current_epoch <= self.args.max_epochs:
            epoch = self.current_epoch
            sys.stdout.flush()

            print(f"\nEPOCH={epoch:03d}/{self.args.max_epochs} - ITER={self.current_iter}")
            # Shuffle training data and vqvae_v1 for n iters
            self.train_n_iters(data_train, loss_type=loss)

            if epoch % self.args.val_freq == 0:
                # Validate the model
                val = self.eval(data_val, loss_type=loss,
                                epoch=epoch,
                                save_to_tboard=self.args.visu_to_tboard)
                # Save ckpt
                if val < self.best_val:
                    self.checkpoint(tag='best_val', extra_dict={'pve': val})
                    self.best_val = val

            if epoch % self.args.ckpt_freq == 0 and epoch > 0:
                self.checkpoint(tag='ckpt_' + str(epoch), extra_dict={'best_val': self.best_val,
                    'best_class': self.best_class})
            if epoch % self.args.restart_ckpt_freq == 0 and epoch > 0:
                    # This one is saved more frequently but erases itself to save memory. Usefull for best-effort models. 
                self.checkpoint(tag='ckpt_restart', extra_dict={'best_val': self.best_val,
                    'best_class': self.best_class})
            self.current_epoch += 1
        return None

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = super(QTrainer, QTrainer).add_trainer_specific_args(parent_parser)
        parser.add_argument("--alpha_root", type=float, default=1)
        parser.add_argument("--alpha_body", type=float, default=1)
        parser.add_argument("--alpha_trans", type=float, default=1)
        parser.add_argument("--alpha_vert", type=float, default=100)
        parser.add_argument("--alpha_fast_vert", type=float, default=0.)
        parser.add_argument("--alpha_codebook", type=float, default=0.25)
        parser.add_argument("--alpha_kl", type=float, default=1.)
        parser.add_argument("--freq_vert", type=int, default=1)
        parser.add_argument("--prop_vert", type=float, default=1.)
        parser.add_argument("--tprop_vert", type=float, default=0.1)
        parser.add_argument("--vert_string", type=str, default=None)


        return parser

def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=400)
    #parser.add_argument("--data_device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dummy_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=5000)
    parser.add_argument("--val_freq", type=int, default=2)
    parser.add_argument("--ckpt_freq", type=int, default=30)
    parser.add_argument("--restart_ckpt_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=2000)

    # Parameters for the classif evaluation
    parser.add_argument("--class_freq", type=int, default=-1)
    parser.add_argument("--fid_freq", type=int, default=20)
    parser.add_argument("--visu_freq", type=int, default=50)
    parser.add_argument("--train_visu_freq", type=int, default=50)
    parser.add_argument("--visu_to_tboard", type=int, default=int(get_user() == 'tlucas'), choices=[0,1])
    parser.add_argument("--n_visu_to_save", type=int, default=2)
    parser.add_argument("--train_data_dir", type=str,
            default='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--val_data_dir", type=str,
            default='data/smplx/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')

    # -train_data_dir  --val_data_dir
    parser.add_argument("--n_train", type=int, default=1000000)
    parser.add_argument("--n_iter_val", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default='logs')
    parser.add_argument("--name", type=str, default='debug')

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-b_train", type=int, default=64)
    parser.add_argument("--val_batch_size", "-b_val", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # parser.add_argument("--bench_w", type=int, default=0)
    # parser.add_argument("--bench_t", type=int, default=0)

    parser.add_argument("--model", type=str, default='conv.Resnet')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])

    # optim parameters
    parser.add_argument("--ab1", type=float, default=0.95, help="Adam beta 1 parameter")
    parser.add_argument("--ab2", type=float, default=0.999, help="Adam beta 2 parameter")

    # Loss choices
    parser.add_argument("--loss", type=str, default='l2', choices=['l2', 'l1', 'laplacian', 'gaussian'])
    parser.add_argument("--l2_verts", type=int, default=0, choices=[0, 1])

    # Print a list of all layers.
    parser.add_argument("--detailed_count", type=int, default=0)

    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vq_seq_len", type=int, default=64)
    parser.add_argument("--num_workers", "-j", type=int, default=16)
    parser.add_argument("--data_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_start", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_classif", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_fid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--class_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seqlen_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--classif_ckpt", type=str, default=None)
    parser.add_argument("--classif_ckpt_babel", type=str, default=None)
    parser.add_argument("--eos_force", type=int, default=1, choices=[0,1])
    parser.add_argument("--use_amp", type=int, default=0, choices=[0, 1])

    script_args, _ = parser.parse_known_args(args)
    Model = {'CausalVQVAE': CausalVQVAE, 'OnlineVQVAE': OnlineVQVAE,
            'TransformerVQVAE': TransformerVQVAE, 'OfflineVQVAE': OfflineVQVAE}[script_args.model]

    parser = QTrainer.add_trainer_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args(args)
    try:
        args.factor = np.prod(args.pool_kernel) # temporal downsampling
    except:
        args.factor = 1

    if args.debug:
        args.num_workers = 1
        args.n_iters_per_epoch = 20
        args.class_freq = 1
        args.fid_freq = 1
        args.val_freq = 1
        args.spin_freq = 1
        args.visu_freq = 1
        args.log_freq = 5
        args.train_batch_size = 23  # Hard to confuse with an other dimension
        args.train_data_dir = 'data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16_nMax1000'
        args.val_data_dir = 'data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16_nMax1000'
        args.dummy = 1  # dummy data
        args.n_iter_val = 2
        if get_user() == 'fbaradel':
            args.n_iters_per_epoch = 1000
            args.train_batch_size = 16

    # Data
    print(f"\nLoading data...")
    loader_train, loader_val = get_data_loaders(args)
    args.type = loader_train.dataset.type

    known_datadirs_to_classifier = {'babel': args.classif_ckpt_babel}
    matching = [k for k in known_datadirs_to_classifier.keys() if k in loader_train.dataset.data_dir]
    args.dataset_type = matching[0] if len(matching) else 'unknown'
    if args.classif_ckpt is None:
        assert len(matching) == 1, "Unknow data dir, provide classif_ckpt manually"
        args.classif_ckpt = known_datadirs_to_classifier[args.dataset_type]

    print(f"Data - N_train={len(loader_train.dataset.pose)} - N_val={len(loader_val.dataset.pose)}")

    # Model
    print(f"\nBuilding the model...")
    print(args)
    in_dim = ((loader_train.dataset.pose[0].size(1) // 3) - 1) * 6 + 3  # jts in 6D repr, trans in 3d coord
    model = Model(in_dim=in_dim, **vars(args)).to(device)
    model.seq_len = args.vq_seq_len

    total_param = print_parameters_count(model, detailed=args.detailed_count)

    reload_epoch = True
    print(f"Number of parameters: {get_parameters(model):,}")
    checkpoint, ckpt_path = get_last_checkpoint(args.save_dir, args.name)
    if checkpoint is None and args.pretrained_ckpt is not None:
        ckpt_path, reload_epoch = args.pretrained_ckpt, False
        checkpoint = torch.load(args.pretrained_ckpt)

    if checkpoint is not None:
        weights = checkpoint['model_state_dict']
        missing, unexpected = model.load_state_dict(weights, strict=False)
        assert not (len(unexpected) or len(missing)), "Problem with loading"
        # Reload centroid counts
        if 'balance_stats' in checkpoint.keys():
            bins = checkpoint['balance_stats']
            model.quantizer.load_state(bins)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.ab1, args.ab2))
    loss_scaler = NativeScaler() if args.use_amp else None

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch, saved_iter = [checkpoint[k] for k in ['epoch', 'iter']] if reload_epoch else [1, 0]
        bv, bc = [checkpoint[k] if k in checkpoint
                else None for k in ['best_val', 'best_class']] if reload_epoch else [None, None]
        print(f"Ckpt succesfully loaded from: {ckpt_path}")
        if 'scaler' in checkpoint:
            assert loss_scaler is not None, "I have found weights for the loss_scaler, but don't have it."
            loss_scaler.load_state_dict(checkpoint['scaler'])
    else:
        epoch, saved_iter = 1, 0
        bv, bc = None, None

    # Trainer
    print(f"\nSetting up the trainer...")
    trainer = QTrainer(model=model, optimizer=optimizer, device=device,
                       args=args, epoch=epoch, start_iter=saved_iter,
                       best_val=bv, best_class=bc, type=loader_train.dataset.type,
                       seq_len=args.seq_len, loss_scaler=loss_scaler)
    
    if args.eval_only:
        # Validate the model; will compute standard fid (without time conditioning), and classification accuracy
        val = trainer.eval(loader_val, loss_type=args.loss, epoch=epoch,
                           save_to_tboard=args.visu_to_tboard)
        print(val)
    else:
        trainer.writer.add_scalar('z_parameter_count', total_param, trainer.current_iter)
        trainer.fit(loader_train, loader_val, loss=args.loss)

if __name__ == "__main__":
    main()
