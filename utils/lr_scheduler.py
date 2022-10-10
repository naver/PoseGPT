# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    def cosval():
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
        return lr
    def apply_scale():
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

    warmup_epochs = args.warmup_epochs or 0
    lr = args.lr * epoch / warmup_epochs if epoch < warmup_epochs else cosval()
    apply_scale()
    return lr
