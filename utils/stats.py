# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.detach()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def class_accuracy(*, logits, target_with_eos):
    pred = torch.argmax(logits, -1)
    valid_ = (1. - (torch.cumsum(((target_with_eos == 0).sum(-1) > 0).float(), dim=-1) > 0).float()).int()
    valid_ = valid_.unsqueeze(-1).repeat(1, 1 , pred.shape[-1])
    acc = 100. * ((target_with_eos == pred).float() * valid_).sum() / valid_.sum()
    return acc
