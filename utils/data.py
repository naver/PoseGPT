# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from torch.utils.data import DataLoader
from dataset.mocap import MocapDataset, worker_init_fn
import torch

def slice_by_index(X, idx):
    filtered = [val for i, val in enumerate(X.split(1, dim=0)) if not bool(idx[i])]
    return torch.cat(filtered, dim=0)

def get_data_loaders(args):
    loader_train = DataLoader(
        MocapDataset(data_dir=args.train_data_dir, seq_len=args.seq_len, training=True,
                     n_iter=args.n_iters_per_epoch * args.train_batch_size, n=-1,
                     data_augment=args.data_augment == 1, dummy=args.dummy == 1,
                     sample_start=args.sample_start == 1
                     ),
        batch_size=args.train_batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)

    niter_val = args.n_iter_val * args.val_batch_size if args.n_iter_val else None
    loader_val = DataLoader(
        MocapDataset(data_dir=args.val_data_dir, seq_len=args.seq_len, training=False, n=-1,
                     dummy=args.dummy == 1, n_iter=niter_val),
        batch_size=args.val_batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=args.n_iter_val is not None)
    return loader_train, loader_val

