# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np

import getpass
import torch.nn.functional as F

RED, ENDC, BLUE = '\033[91m', '\033[0m', '\033[94m'
red = lambda string: RED + string + ENDC
blue = lambda string: BLUE + string + ENDC


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_user():
    return getpass.getuser()
