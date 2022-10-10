# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import roma

from threed.skeleton import get_smplx_pose, get_smplh_pose, get_smpl_pose

def pose_to_vertices(y, pose_type, alpha, bm, parallel=False,
        manual_seq_len=None):
    """
    Map SMPL-H/X parameters to a human mesh
    Args:
        - y: [batch_size, seq_len, K]
    Return:
        - verts: [batch_size, seq_len, n_vertices, 3]
    """
    batch_size, seq_len, *_ = y.size()

    if not parallel:
        list_verts = []
        for i in range(batch_size):
            pose = eval(f"get_{pose_type}_pose")(y[i])
            if alpha == 0:
                if 'trans' in pose:
                    pose['trans'] *= 0.
                elif 'transl' in pose:
                    pose['transl'] *= 0.
            # for k,v in pose.items():
                # print(k+':'+str(v.shape))
            if manual_seq_len is not None:
                pose.update({'betas': torch.zeros((manual_seq_len,10)).cuda(),
                             'batch_size': manual_seq_len})
            verts = bm(**pose).vertices
            list_verts.append(verts)
        verts = torch.stack(list_verts)
        return verts
    else:
        cy = y.reshape((-1, y.shape[-1]))
        pose = eval(f"get_{pose_type}_pose")(cy)
        if alpha == 0:
            if 'trans' in pose:
                pose['trans'] *= 0.
            elif 'transl' in pose:
                pose['transl'] *= 0.
        # for k,v in pose.items():
            # print(k+':'+str(v.shape))
        if manual_seq_len is not None:
            pose.update({'betas': torch.zeros((manual_seq_len,10)).cuda(),
                             'batch_size': manual_seq_len})
        verts = bm(**pose).vertices
        verts = verts.reshape((y.shape[0], y.shape[1], verts.shape[-2], verts.shape[-1]))
        return verts

def get_trans(delta, valid):
    """ Compute absolute translation coordinates from deltas """
    trans = [delta[:, 0].clone()]
    for i in range(1, delta.size(1)):
        if valid is not None:
            assert valid.shape[1] > i, "Here is a bug"
            d = delta[:, i] * valid[:, [i]].float()
        else:
            d = delta[:, i]
        trans.append(trans[-1] + d)
    trans = torch.stack(trans, 1)
    return trans


def six_dim(x):
    """ Move to 6d representation and represent translations as deltas """
    batch_size, seq_len, _ = x.size()

    # move to 6d representation
    x = x.reshape(batch_size, seq_len, -1, 3)
    trans = x[:, :, -1]  # [batch_size,seq_len,3]
    rotvec = x[:, :, :-1]
    rotmat = roma.rotvec_to_rotmat(rotvec)  # [batch_size,seq_len,n_jts,3,3]
    x = torch.cat([rotmat[..., :2].flatten(2), trans], -1)  # [batch_size,seq_len,n_jts*6+3]
    return x, rotvec, rotmat, trans

class SimplePreparator(nn.Module):
    def __init__(self, mask_trans, pred_trans, **kwargs):
        super(SimplePreparator, self).__init__()
        self.mask_trans = mask_trans
        self.pred_trans = pred_trans

    def forward(self, x, **kwargs):
        x, rotvec, rotmat, trans = six_dim(x)
        assert self.mask_trans == (not self.pred_trans), "mask_trans and pred_trans incoherent"
        if self.mask_trans:
            trans = torch.zeros_like(trans)
        trans_delta = trans[:, 1:] - trans[:, :-1]
        return x, rotvec, rotmat, trans, trans_delta, None
