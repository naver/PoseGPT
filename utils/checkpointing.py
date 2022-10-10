# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch

def get_last_checkpoint(save_dir, name):
    """ Find the last checkpoint available in the experiment folder """
    ckpt_dir = os.path.join(save_dir, name, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, None

    restart_ckpt = os.path.join(ckpt_dir, 'ckpt_restart.pt')
    if os.path.isfile(restart_ckpt):
        return torch.load(restart_ckpt), 'ckpt_restart.pt'
    else:
        ckpts = sorted([c for c in os.listdir(ckpt_dir) if 'ckpt_' in c],
                       key=lambda s: int(s.split('ckpt_')[1].split('.')[0]))
        if len(ckpts) > 0:
            return torch.load(os.path.join(ckpt_dir, ckpts[-1])), ckpts[-1]
    return None, None


