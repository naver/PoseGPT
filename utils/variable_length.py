# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

""" Utils to deal with the fact that sequences are of varying length """

def valid_concat_rot_trans(rot, trans, valid, squeeze=True):
    """ Concatenate rotation and translation, mask invalid part by repeating last valid."""
    concat = torch.cat((rot.flatten(2), trans.flatten(2)), dim=-1)
    concat = repeat_last_valid(concat, valid)
    if squeeze:
        poses = [e.squeeze(0).clone().cpu() for e in torch.split(concat, 1, dim=0)]
    else:
        poses = concat
    return poses

def unsqueeze_mask_to_shape(x, shape):
    assert len(shape) in [3, 4], "unexpected shape"
    return x.unsqueeze(-1).unsqueeze(-1) if len(shape) == 4 else x.unsqueeze(-1)

def select_last_valid(x, v):
    """ For an input of shape [batch_size, temporal_dim, ...] and valid of shape [batch_size, tdim], 
    selects the last slice of x along the temporal dimension that has a valid mask. """
    # Read valid from the right: cumsum finds the first 1, then reverse it again.
    last_valid_mask = (torch.cumsum(v.flip(dims=(-1,)).int(), dim=-1) == 1).int().flip(dims=(-1,))
    assert all(last_valid_mask.sum(-1) <= 1), "Unexpected behaviour"
    lvm = unsqueeze_mask_to_shape(last_valid_mask, x.shape)
    last_valid = (x * lvm).sum(1, keepdim=True)
    return last_valid

def repeat_last_valid(x, v):
    """ Select last valid parameter slice and repeat it to cover the invalid part,
    to have constant dimensions. """
    # Get last valid 
    pad = select_last_valid(x, v)
    uv = unsqueeze_mask_to_shape(v, x.shape)
    return x * uv + pad * (1 - uv.int())


def slice_last_valid(valid, output):
    """ Input shape: b, T, c
    Take the last valid slice at t <= T for each batch element.
    """
    max_len = torch.cumsum(valid, dim=-1)[:, -1]
    split_batch = torch.split(output, split_size_or_sections=1, dim=0)
    slices = [e[:, (max_len - 1)[i], ...] for i, e in enumerate(split_batch)]
    out = torch.cat(slices, dim=0)
    return out

