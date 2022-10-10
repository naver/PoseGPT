# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.mocap import MocapDataset, worker_init_fn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.ae_utils import red
import pickle as pkl
from models.classifiers import Autoreg_classifier, TransformerDiscriminator
from classify import train_classifier, evaluate_classifier
from utils.variable_length import valid_concat_rot_trans

from utils.fid import (calculate_activation_statistics, calculate_diversity_multimodality, calculate_fid, multiclass_div_mod)
import smplx
from functools import partial
from utils.utils import compute_map


def filter_empty(li, valid):
    """ If valid is all zeros, remove the elements in the lists in li at the corresponding index """
    return [[a[i] for i, v in enumerate(valid) if v[0]] for a in li]

@torch.no_grad()
def extract_activations(model, data_loader, device):
    """ Forward pass with a classifier for each element of the batch, accumulate the features. """
    acts, accuracies = [], []
    for x, valid, y in tqdm(data_loader):
        x, valid = x.to(device), valid.to(device)
        act = model.forward_fid(x, valid)
        acts.append(act.reshape(act.shape[0], -1))
        # Compute acccuracy
        y, y_hat = y.to(device), model.forward(x, valid)
        accuracies.append((torch.argmax(y, 1) == torch.argmax(y_hat, 1)).float())
    accuracies = np.float(100. * torch.cat(accuracies).mean().cpu())
    return torch.cat(acts, dim=0), accuracies


def precompute_real_fid(train_data_dir, seq_len, class_model, model_name):
    """  Extract activations on the dataset, compute mean and var, dump it"""
    data_loader = DataLoader(MocapDataset(data_dir=train_data_dir, seq_len=seq_len, training=False,
                                          n_iter=None, n=-1, data_augment=0),
                              batch_size=32, num_workers=1, prefetch_factor=2, shuffle=False,
                              worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activations, accuracy = extract_activations(class_model, data_loader, device)
    print(f"Accuracy={accuracy:.2f}")
    mu, sigma = calculate_activation_statistics(activations)
    with open(f'./logs/{model_name}/mu_sigma', 'wb') as f:
        pkl.dump({'mu': mu, 'sigma': sigma}, f)
    return mu, sigma


def compute_class_accuracy(data_loader, device, action_model_ckpt, data_type=None):
    """ Evaluate pretrained classifier on samples using the conditioning actions as target."""
    action_model, bm = build_and_load_classifier(action_model_ckpt, data_loader,
                                        device, data_type=data_type)
    fc = partial(forward_classifier, bm=bm, classifier=action_model,
                 device=device, cat_out=True)
    mAP = evaluate_classifier(data_loader, classifier=fc, do_logging=False,
                              writer=None, current_iter=None, device=device, data_type=data_type)
    return mAP

def build_classifier(device, data_type):
    if 'babel' in data_type:
        cut_joints = 0 
        # if use_bm:
        #input_dim = 381 if not cut_joints else 3 * cut_joints
        classifier = TransformerDiscriminator(device=device, in_dim=168).cuda()
        #classifier = TransformerDiscriminator(smpl_input=False, cut_joints=24).cuda()
    else:
        raise NotImplementedError("Unknown data_dir, which model should I use?")
    classifier.method = {'babel': 'TD'}[data_type]
    return classifier

def build_and_load_classifier(action_model_ckpt, data_loader, device, data_type=None,
                              use_bm=0):
    if data_type is None:
        matches = [k for k in ['babel'] if k in data_loader.dataset.data_dir]
        if len(matches) != 1:
            raise NotImplementedError("Unknown data_dir, which model should I use?")
        data_type = matches[0]

    classifier = build_classifier(device, data_type)
    checkpoint = torch.load(action_model_ckpt)
    if data_type == 'babel':
        weights = {k.replace('classifier.', ''): v for k,v in checkpoint['model_state_dict'].items()}
        weights = {k: v for k, v in weights.items() if 'bm' not in k}
        classifier.load_state_dict(weights)
    else:
        raise NotImplementedError("Unknown data_dir, which model should I use?")
    classifier.eval()

    # Define how to forward the classifier (if a body model is involved, we need a for loop on the batch to avoid going OOM).
    if isinstance(classifier, Autoreg_classifier) or isinstance(classifier, TransformerDiscriminator):
        bm = None
    else:
        raise NotImplementedError("No definition of classifier forward")
    if use_bm:
        raise NotImplementedError("Need to implement joints based evaluation.")
    return classifier, bm

def forward_classifier(x, valid, classifier, bm, device, cat_out=False):
    """ When using a body model, sequentially forward each element in the batch and zip results together.
        Otherwise, simply make a batched forward and split the result. """

    if isinstance(classifier, Autoreg_classifier) or isinstance(classifier, TransformerDiscriminator):
        #def forward_classifier(x, valid):
        # TODO we could chose to wrap with body model.
        res = classifier.forward_fid(x.to(device), valid.to(device))
        if not cat_out:
            return torch.split(res[0], 1, dim=0), torch.split(res[1], 1, dim=0)
        return res[0], res[1]
    else:
        raise NotImplementedError("No classifier forward defined")

def get_real_fid_path(action_model_ckpt):
    real_fid_path = os.path.join('/'.join(action_model_ckpt.split('/')[:-3]), 'real_fid_stats.pkl')
    return real_fid_path


@torch.no_grad()
def sample_pose_dataset(pose_gpt, path, data_loader, preparator, device,
                        class_conditional=True, seqlen_conditional=True,
                        temperature=1.0, top_k=None, cond_steps=0):
    """ Loop over the data_loader; take the class embeddings and sample new poses with the same embeddings."""
    os.makedirs(path, exist_ok=True)
    if os.path.isfile(os.path.join(path, 'pose.pkl')):
        print("Dataset with this name already sampled.")
        return path

    zidx = None
    with torch.no_grad():
        samples = []
        labels = []
        print(red("> Extracting activations for fid..."))
        for x, valid, actions in tqdm(data_loader):
            x, valid, actions = x.to(device), valid.to(device), actions.to(device)
            x, *_ = preparator(x) # Correct input format for the model
            _valid = valid
            seqlens = valid.sum(1)

            (rot, trans), valid, zidx = pose_gpt.sample_poses(zidx, x, valid,
                    actions=actions if class_conditional else None,
                    seqlens=seqlens if seqlen_conditional else None,
                    temperature=temperature,
                    top_k=top_k,
                    cond_steps=cond_steps, return_zidx=True)
            sample_valid = _valid if pose_gpt.sample_eos_force else valid
            # Concatenate rotation and translation.
            poses = valid_concat_rot_trans(rot, trans, sample_valid)
            samples.extend(poses)
            labels.extend([e.squeeze(0).clone().cpu() for e in actions.split(1, dim=0)])

        torch.save(labels, os.path.join(path, 'action.pt'))
        f = open(os.path.join(path, 'pose.pkl'), 'wb')
        pkl.dump(samples, f)
        print("OK!")

@torch.no_grad()
def extract_from_classifier(*, batch_sampler, data_loader, preparator, device, action_model_ckpt,
                            debug=False, real_fid_path, sample_options={}, summarize=False, dump=False):

    classifier, bm = build_and_load_classifier(action_model_ckpt, data_loader, device)
    fc = partial(forward_classifier, bm=bm, classifier=classifier, device=device)
    real_fid_path = get_real_fid_path(action_model_ckpt)
    evaluate_real = (not os.path.isfile(real_fid_path)) or debug

    zidx_or_bs = None
    acc_gt, acc_gen, act_gt, act_gen, labels = [], [], [], [], []
    with torch.no_grad():
        print(red("> Generating synthetic dataset ..."))
        for i, (x, valid, actions) in enumerate(tqdm(data_loader)):
            poses, sample_valid, zidx_or_bs = batch_sampler(x=x, valid=valid, preparator=preparator,
                actions=actions, device=device, zidx_or_bs=zidx_or_bs, data_loader=data_loader, i=i, **sample_options)

            y_gen, fgen = filter_empty(fc(poses, sample_valid), sample_valid)
            actions = torch.cat(filter_empty([torch.split(actions, 1, dim=0)], sample_valid)[0], dim=0)
            act_gen.extend([f for f in fgen if not torch.isnan(f).int().sum().bool()])

            labels.extend(actions)
            mAP = compute_map(actions, y_gen)
            acc_gen.append(mAP)
            if evaluate_real:
                y_gt, fgt = filter_empty(fc(x.to(device), valid.to(device)), sample_valid)
                act_gt.extend(fgt)
                mAP_real = compute_map(actions, y_gt)
                acc_gt.append(mAP_real)
    if not summarize:
        return actions, labels, act_gt, act_gen, acc_gen
    return metrics(actions, labels, act_gt, act_gen,
                     acc_gen, real_fid_path, debug, dump=dump)

@torch.no_grad()
def metrics(actions, labels, act_gt, act_gen,
               acc_gen, real_fid_path,
               debug, dump):

    evaluate_real = (not os.path.isfile(real_fid_path)) or debug
    num_classes = actions.shape[1]
    feats_gen = torch.cat(act_gen)
    div_gen, mod_gen = multiclass_div_mod(feats_gen, labels, nb_classes=num_classes)
    stats_gen = calculate_activation_statistics(feats_gen)

    if evaluate_real:
        feats_gt = torch.cat(act_gt)
        div_gt, mod_gt = multiclass_div_mod(feats_gt, labels, nb_classes=num_classes)
        stats_gt = calculate_activation_statistics(feats_gt)
        if dump:
            with open(real_fid_path, 'wb') as f:
                pkl.dump({'div_gt': div_gt, 'mod_gt': mod_gt,
                          'fid_stats': stats_gt}, f)
    else:
        with open(real_fid_path, 'rb') as f:
            stats = pkl.load(f)
            div_gt, mod_gt, stats_gt= stats['div_gt'], stats['mod_gt'], stats['fid_stats']
    try:
        fid = float(calculate_fid(stats_gt, stats_gen))
    except ValueError as e:
        if debug:
            print("FID has produced NaNs, that's fine as you are in debug mode")
            fid = 100.
        else:
            raise ValueError(e)

    print(f"Gen   : FID:{fid:.2f} - Acc/mAP: {100. * np.asarray(acc_gen).mean():.1f} - Div:{div_gen:.2f} - Multimod:{mod_gen:.2f} ")
    return {'fid': fid, 'acc': 100. * np.asarray(acc_gen).mean(), 'diversity': div_gen, 'multimodality': mod_gen}


def classification_evaluation(model, data_loader, log_dir, epoch, args, action_model_ckpt,
                              preparator, while_training=True, debug=False, data_type=None,
                              temperature=1.0, top_k=None, cond_steps=0):
    """ A classifier trained on real data is evaluated on samples and vice-versa. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples_path = os.path.join(log_dir, 'samples' + str(epoch) + '_t_' + str(temperature)  + '_top' + str(top_k) + '_cond' + str(cond_steps))
    is_training = model.gpt.training
    model.gpt.eval()
    sample_pose_dataset(model, samples_path, data_loader, preparator=preparator,
                                      device=device, class_conditional=args.class_conditional,
                                      seqlen_conditional=args.seqlen_conditional)
    if is_training:
        model.gpt.train()
    sample_data_loader = DataLoader(MocapDataset(data_dir='./' + samples_path, seq_len=args.seq_len, training=False,
                                                 n_iter=None, n=-1, data_augment=0, dummy=0),
                                    batch_size=32, num_workers=1, prefetch_factor=2, shuffle=False,
                                    worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)
    writer = SummaryWriter(log_dir)
    # Evaluate a classifier pretrained with real data on samples.
    mAP = compute_class_accuracy(data_loader=sample_data_loader, device=device,
                                 action_model_ckpt=action_model_ckpt, data_type=data_type)
    train_tag = 'at_train' if while_training else 'at_test'
    writer.add_scalar(f"class_acc_samples/{train_tag}", mAP, epoch)

    with open(os.path.join(samples_path, 'sample_classif_accuracy.txt'), 'a') as f:
        f.write('Sample classification accuracy: ' + str(mAP))

    #### Train a classifier on generated data and evaluate it on real validation data.
    # Generated Actions: (60,)
    val_data_loader = DataLoader(MocapDataset(data_dir=args.val_data_dir, seq_len=args.seq_len, training=False,
                                              n=args.overfit, dummy=args.dummy == 1),
                                 batch_size=args.val_batch_size, num_workers=args.num_workers,
                                 prefetch_factor=args.prefetch_factor, shuffle=True,
                                 worker_init_fn=worker_init_fn, pin_memory=False, drop_last=False)

    if debug:
        # Options set to debug 
        opts = {'ckpt_freq':100, 'restart_ckpt_freq': 500, # So never.
                'val_freq': 1, 'log_every_n_iter': 50, 'n_iters_per_epoch' :100,
                'max_epochs': 5, 'learning_rate': 2e-4}
    elif while_training:
        # Options set to train very fast 
        opts = {'ckpt_freq':100, 'restart_ckpt_freq': 500, # So never.
                'val_freq': 5, 'log_every_n_iter' :200, 'n_iters_per_epoch' :1000,
                'max_epochs' :150, 'learning_rate': 2e-4}

    else:
        opts = {'val_freq': 10, 'ckpt_freq':100, 'restart_ckpt_freq': 50,
                'log_every_n_iter' :200, 'n_iters_per_epoch' :1000, 'max_epochs' :500,
                'learning_rate': 5e-5}

    classifier = build_classifier(device, data_type)
    best_val_mAP = train_classifier(device=device, save_dir='/'.join(samples_path.split('/')[:-1]),
                                    name=samples_path.split('/')[-1],
                                    classifier=classifier,
                                    loader_train=sample_data_loader, loader_val=val_data_loader,
                                    train_batch_size=32, val_batch_size=32, do_logging=True,
                                    extra_log_tag='_reals', data_type=data_type, **opts)
    return best_val_mAP


if __name__ == '__main__':
    print("Hello there!")
