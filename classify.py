# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from argparse import ArgumentParser
import warnings
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
from utils.stats import AverageMeter
from utils.checkpointing import get_last_checkpoint
from utils.ae_utils import red
from utils.data import get_data_loaders
from sklearn.metrics import average_precision_score as AP

from models.classifiers import TransformerDiscriminator, bm_classifier
from utils.param_count import print_parameters_count
import smplx
from utils.constants import SMPLX_DIR
from threed.skeleton import get_smplx_pose, get_smplh_pose, get_smpl_pose

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class CTrainer(nn.Module):
    def __init__(self, *, save_dir, name, learning_rate, log_every_n_iter,
                 val_freq, ckpt_freq, restart_ckpt_freq, n_iters_per_epoch,
                 max_epochs, device, classifier, do_logging=True,
                 extra_log_tag=None, data_type=None, ckpt_num=''):

        super(CTrainer, self).__init__()
        self.device = device
        self.extra_log_tag = extra_log_tag
        self.classifier = classifier

        self.classifier.to(device)

        multi = data_type is not None and ('humanact' in data_type or 'grab' in data_type)
        self.criterion = nn.CrossEntropyLoss() if multi else nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.save_dir, self.name = save_dir, name

        self.log_every_n_iter = min(log_every_n_iter, n_iters_per_epoch)
        self.do_logging = do_logging

        self.val_freq, self.ckpt_freq = val_freq, ckpt_freq
        self.restart_ckpt_freq = restart_ckpt_freq

        self.n_iters_per_epoch, self.max_epochs = n_iters_per_epoch, max_epochs

        self.ckpt_num = ckpt_num
        self.classif_method = classifier.method

        # Loading weights if they are available
        self.init_logdir()
        checkpoint, ckpt_path = get_last_checkpoint(save_dir, name + '/classification_' + self.classif_method)
        if checkpoint is not None:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.current_iter, self.current_epoch = 0, 0

    def init_logdir(self):
        """ Create the log dir space and associated subdirs """
        log_dir_root = os.path.join(self.save_dir, self.name, 'classification_' + self.classif_method)
        if len(self.ckpt_num) > 0:
            log_dir_root = os.path.join(log_dir_root, self.ckpt_num)
        self.log_dir = log_dir_root
        os.makedirs(log_dir_root, exist_ok=True)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.do_logging:
            self.writer = SummaryWriter(self.log_dir)

    def save_ckpt(self, name, epoch, extra_dict={}):
        weights = {k: v for k,v in self.classifier.state_dict().items() if not 'bm.' in k}
        to_save = {'iter': self.current_iter, 'model_state_dict': weights,
                   'optimizer_state_dict': self.optimizer.state_dict(), 'epoch': epoch}
        to_save.update(extra_dict)
        torch.save(to_save, os.path.join(self.ckpt_dir, name))


    def fit(self, data_train, data_val):
        """
        Train and evaluate a model using training and validation data
        """
        best_val = -1e5
        self.current_iter = 0
        self.current_epoch = 0
        for epoch in range(1, self.max_epochs):
            sys.stdout.flush()
            print(f"\nEPOCH={epoch:03d}/{self.max_epochs} - ITER={self.current_iter}")

            self.train_n_iters(data_train)

            # Validation and checkpointing
            if epoch % self.val_freq == 0 and (epoch > 0 or self.val_freq == 1):
                val = self.evaluate(data_val)
                if val > best_val:
                    self.save_ckpt('best.pt', epoch, {'pve': val})
                    best_val = val.item()
            if epoch % self.ckpt_freq == 0 and (epoch > 0 or self.ckpt_freq == 1):
                self.save_ckpt("ckpt_" + str(epoch) + ".pt", epoch)
            if epoch % self.restart_ckpt_freq == 0 and (epoch > 0 or self.restart_ckpt_freq == 1):
                self.save_ckpt("ckpt_restart.pt", epoch)
            self.current_epoch += 1
        return best_val

    def log_train_stats(self, bce):
        if self.do_logging:
            for k, v in zip(["train/bce"], [bce.avg]):
                self.writer.add_scalar(k, v, self.current_iter)

    def train_n_iters(self, data_train):
        """ Train for one epoch """

        self.train()
        bce, data_time, batch_time = [AverageMeter(k, ':6.3f') for k in ['Bce', 'data_time', 'batch_time']]
        end = time.time()
        print(red("> Training classifier ..."))
        for x, valid, y in tqdm(data_train):
            x, valid, y = x.to(self.device), valid.to(self.device), y.to(self.device)
            data_time.update(time.time() - end)
            logits = self.classifier(x, valid)
            assert len(logits.shape) == 2, "unexpected shape"
            target = torch.nonzero(y)[:, 1].long() if isinstance(self.criterion, nn.CrossEntropyLoss) else y.float()
            loss = self.criterion(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            bce.update(loss)
            self.current_iter += 1

            if not self.current_iter % self.log_every_n_iter:
                self.log_train_stats(bce)

        print(f"TRAIN:")
        self.log_train_stats(bce)

        self.writer.add_scalar(f"gpu_load/batch_time", batch_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/it_per_sec", 1. / batch_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/data_time", data_time.avg, self.current_iter)

    def evaluate(self, data_val):
        self.eval()
        self.classifier.eval()
        return evaluate_classifier(data_val, self.classifier, self.do_logging,
                                   self.writer, self.current_iter, self.device, self.extra_log_tag)

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
        parser.add_argument("--train_batch_size", "-b_train", type=int, default=32)
        parser.add_argument("--val_batch_size", "-b_val", type=int, default=32)
        return parser

@torch.no_grad()
def evaluate_classifier(data_val, classifier, do_logging, writer,
                        current_iter, device, extra_log_tag=None,
                        data_type=None):
    """ Given a classifier and some data, compute the accuracy / mAP."""
    print(red("> Evaluating classifier ..."))
    multi_class = data_type is None or 'babel' in data_type
    targets, scores = [], []

    for x, valid, y in tqdm(data_val):
        x, valid, y = x.to(device), valid.to(device), y.to(device)
        output = classifier(x, valid)
        logits = output if not isinstance(output, (list, tuple)) else output[0]
        batch_scores = torch.sigmoid(logits) if multi_class else torch.nn.functional.softmax(logits)
        scores.append(batch_scores.cpu())
        targets.append(y.cpu())
    scores, targets = [torch.cat(x).numpy() for x in [scores, targets]]

    # average precision for each class
    APs = [np.nan_to_num(AP(targets[:, k], scores[:, k])) for k in range(targets.shape[1])]
    mAP = np.asarray(APs).mean() * 100.

    if do_logging:
        tag = "val_mAP" + (extra_log_tag if extra_log_tag is not None else '')
        writer.add_scalar(tag, mAP, current_iter)

    print(f"VAL:")
    print(f"    - mAP: {mAP:.3f}")

    return mAP

def build_AR(mlp_dim, nb_classes, smpl_input=True, gru_dim=256, **kwargs):
    input_dim = 168 if smpl_input else 381
    return Autoreg_classifier(input_dim=input_dim, gru_dim=gru_dim,
                              mlp_dim=mlp_dim, nb_classes=nb_classes)

def build_TR(device, smpl_input=True, cut_joints=None):
    input_dim = 168 if smpl_input else (381 if not cut_joints else 3 * cut_joints)
    return TransformerDiscriminator(device=device, in_dim=input_dim)

def build_classifier(args, data_type, device, smpl_input=True, cut_joints=None):
    if args.classif_method == 'TR':
        classifier = build_TR(device, smpl_input=smpl_input, cut_joints=cut_joints)
    else:
        raise NotImplementedError("Unknown classifier")
    classifier.method = args.classif_method
    return classifier

def train_classifier(*, device, save_dir, name, loader_train,
                     classifier, loader_val, learning_rate=1e-5, val_freq=5,
                     ckpt_freq=50, restart_ckpt_freq=25,
                     log_every_n_iter=-1, n_iters_per_epoch=1000,
                     max_epochs=1000000,
                     do_logging=True, extra_log_tag=None,
                     data_type=None, **kwargs):
    # Trainer
    print(f"\nSetting up the trainer for action recognition...")
    trainer = CTrainer(classifier=classifier, save_dir=save_dir, name=name, learning_rate=learning_rate,
                      log_every_n_iter=log_every_n_iter, val_freq=val_freq, ckpt_freq=ckpt_freq,
                      restart_ckpt_freq=restart_ckpt_freq, n_iters_per_epoch=n_iters_per_epoch,
                      max_epochs=max_epochs, device=device, do_logging=do_logging, extra_log_tag=extra_log_tag,
                      data_type=data_type)
    
    if 'pretrained_ckpt' not in list(kwargs.keys()):
        kwargs['pretrained_ckpt']  = None
    if 'eval_only' not in list(kwargs.keys()):
        kwargs['eval_only'] = False

    if kwargs['pretrained_ckpt'] is not None:
        checkpoint = torch.load(kwargs['pretrained_ckpt'])
        missing, unexpected = trainer.classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        assert len(missing) == 0, "Missing keys"
        assert len(unexpected) == 0, "Unexpected keys"

    if kwargs['eval_only']:
        return trainer.evaluate(data_val=loader_val)
    else:
        return trainer.fit(data_train=loader_train, data_val=loader_val)


def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=150)
    #parser.add_argument("--data_device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=1000)
    parser.add_argument("--num_workers", "-j", type=int, default=16)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--data_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--sample_start", type=int, default=1, choices=[0, 1])

    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_every_n_iter", "-log_it", type=int, default=400)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--ckpt_freq", type=int, default=300)
    parser.add_argument("--restart_ckpt_freq", type=int, default=50)
    parser.add_argument("--train_data_dir", type=str,
                        default='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--val_data_dir", type=str,
                        default='data/smplx/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--save_dir", type=str, default='logs')
    parser.add_argument("--name", type=str, default='debug_eval')
    parser.add_argument("--n_iter_val", type=int, default=None)

    # Classifier parameters
    parser.add_argument("--mlp_dim", type=int, default=120)
    parser.add_argument("--gru_dim", type=int, default=256)
    parser.add_argument("--use_last", type=int, default=1, choices=[0,1])
    parser.add_argument("--classif_method", type=str, default='AR', choices=['AR', 'AMD', 'TR'])
    parser.add_argument("--use_bm", type=int, default=0, choices=[0,1])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--joints_parallel", type=int, default=1, choices=[0, 1])
    parser.add_argument("--proportion", type=float, default=0.2)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cut_joints", type=int, default=None)
    parser.add_argument("--center_at_pelvis", type=int, default=1, choices=[0,1])

    parser = CTrainer.add_trainer_specific_args(parser)
    args = parser.parse_args(args)
    params_root = os.path.join(args.save_dir, args.name, 'classification_' + args.classif_method)
    os.makedirs(params_root, exist_ok=True)

    with open(os.path.join(params_root, 'hparams.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    loader_train, loader_val = get_data_loaders(args)

    data_types = ['humanact', 'babel', 'grab']
    matching = [k for k in data_types if k in loader_train.dataset.data_dir]
    assert len(matching) == 1, "Unknown data type"
    data_type = matching[0]
    classifier = build_classifier(args, data_type, device,
                                  smpl_input=not args.use_bm,
                                  cut_joints=args.cut_joints)

    if args.use_bm:
        classifier = bm_classifier(classifier, time_dim=64, proportion=args.proportion,
                                   batch_size=32, device=device, cut_joints=args.cut_joints,
                                   center_at_pelvis=args.center_at_pelvis)

    print_parameters_count(classifier, detailed=0)
    train_classifier(classifier=classifier, device=device, loader_train=loader_train,
                     loader_val=loader_val, data_type=data_type, **vars(args))


if __name__ == "__main__":
    main()

