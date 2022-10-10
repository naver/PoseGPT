# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn.functional as F
import numpy as np

from functools import reduce


def l2(x_hat, x, device):
    return torch.zeros((1), requires_grad=False).to(device) if x_hat is None else F.mse_loss(x_hat, x, reduction='none')

def l1(x_hat, x, device):
    return torch.zeros((1), requires_grad=False).to(device) if x_hat is None else torch.abs(x_hat - x)

def gaussian_nll(x_tilde, x, log_sigma):
    """ Negative log-likelihood of an isotropic Gaussian density """
    if x_tilde is None:
        res = torch.zeros((1), requires_grad=False).to(log_sigma.device)
        return {'energy': res, 'nll': res, 'norm': res}
    # Product of all dimensions but the first two.
    # N = reduce((lambda x, y: x * y), [x.shape[d] for d in range(1, len(x.shape))])
    # NOTE:  Log norm is no longer multiplied by N (nb of dimensions),
    # as it is now broadcasted and summed.
    log_norm = - 0.5 * (np.log(2 * np.pi) + log_sigma)
    log_energy = - 0.5 * F.mse_loss(x_tilde, x, reduction='none') / torch.exp(log_sigma)
    return {'energy': -log_energy, 'nll': - (log_norm + log_energy), 'norm': -log_norm}

def laplacian_nll(x_tilde, x, log_sigma):
    """ Negative log likelihood of an isotropic Laplacian density """
    if x_tilde is None:
        res = torch.zeros((1), requires_grad=False).to(log_sigma.device)
        return {'energy': res, 'nll': res, 'norm': res}
    log_norm = - (np.log(2) + log_sigma)
    log_energy = - (torch.abs(x_tilde - x)) / torch.exp(log_sigma)
    #return - (log_norm + log_energy).sum([d for d in range(2, len(x.shape))])
    return {'energy': -log_energy, 'nll': - (log_norm + log_energy), 'norm': -log_norm}

def compute_elbo_discrete(nlls, *, idx_shape, dicsize):
    log_prob = - torch.stack(nlls).sum(0)
    kl = np.log(dicsize) * (idx_shape[1] * idx_shape[2] / log_prob.shape[-1])
    elbo = log_prob - kl # the cost is divided between by the number of features predicted.
    return elbo.mean(), kl.mean()

