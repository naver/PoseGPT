# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from sklearn.metrics import average_precision_score
from functools import reduce
import numpy as np
import torch

import argparse

def valid_reduce(x, mask, reduction):
    """ Multiply x by mask and reduce the right way. """

    assert reduction in ['mean', 'sum'], "Unknown reduction"
    if len(x.shape) == 1 and x.shape[0] == 1: # Just a scalar
        return x.sum() if reduction == 'sum' else x.mean()

    # Unsqueeze mask to be of the right shape 
    for _ in range(0, len(x.shape) - len(mask.shape)): # Prepare valid broadcasting
        mask = mask.unsqueeze(-1)

    if reduction == 'mean':
        # We can't simply do .mean() as masked scalars would decrease the average,
        # so we sum and divide by the number of non-masked scalars.

        # Number of scalars to divide by.
        tdim = mask.float().mean([d for d in range(2, len(x.shape))]).sum(-1)

        # Avoid division by 0; if tdim was 0, it will be masked anyway so we don't care.
        tdim = (tdim * (tdim != 0).float() + (tdim == 0).float())
        return (mask.float() * x).mean([d for d in range(2, len(x.shape))]).sum(-1) / tdim

    return (mask.float() * x).sum([d for d in range(1, len(x.shape))])

def count_dim(a):
    return reduce((lambda x, y: x * y), [a.shape[d] for d in range(1, len(a.shape))])
    
def subsamble_random_offset(bs, period, tdim, var_list):
    assert bs > 0, "Nothing to train vertices on"
    offset = np.random.randint(0, period)
    subsample = lambda x: x[:bs, offset::period, ...][:, :tdim, ...]
    return [subsample(x) for x in var_list]

def compute_map(actions, y_gen_logits):
    y_scores = torch.cat(y_gen_logits)
    y_scores = torch.sigmoid(y_scores).cpu().numpy()
    y = actions.cpu().numpy()
    list_average_precision = []
    for k in range(y.shape[1]):
        val = average_precision_score(y[:, k], y_scores[:, k])
        val = np.nan_to_num(val)
        list_average_precision.append(val)
    # NOTE: we leave it as a score between 0 and 1 to be coherent with the way accuracy is handled
    mAP = np.asarray(list_average_precision).mean()
    return mAP


