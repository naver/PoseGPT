# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
from prettytable import PrettyTable

RED, ENDC, BLUE = '\033[91m', '\033[0m', '\033[94m'
red = lambda string: RED + string + ENDC
blue = lambda string: BLUE + string + ENDC

def magnitude(n):
    tags = ['', ' K', red(' M'), red(' B'), red(' T')]
    n = float(n)
    millidx = max(0, min(len(tags) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return ' {:.0f}{}'.format(n / 10 ** (3 * millidx), tags[millidx]) if millidx > 0 else '-'

def print_parameters_count(model, detailed=False, tag=''):
    """ Print number of parameters in each module"""

    total_params_train, total_params_freeze = 0, 0
    if detailed:
        table = PrettyTable([tag + "Layers", "Trainable params", "Frozen params", "Magnitude"])
        for name, parameter in model.named_parameters():
            param = parameter.numel()
            if parameter.requires_grad:
                table.add_row([name, param, 0, magnitude(param)])
                total_params_train += param
            else:
                table.add_row([name, 0, param, magnitude(param)])
                total_params_freeze += param
        table.add_row(['----', '-----', '-----', '-----'])
        table.add_row(['Total', total_params_train, total_params_freeze,
                       magnitude(total_params_train + total_params_freeze)])

    else:
        table = PrettyTable([tag + "Modules", "Trainable params", "Frozen params", "Magnitude"])
        for name, m in model.named_children():
            count_params_t, count_params_f = 0, 0
            for parameter in m.parameters():
                param = parameter.numel()
                if parameter.requires_grad:
                    count_params_t += param
                else:
                    count_params_f += param
            if count_params_t + count_params_f > 0:
                table.add_row([name, count_params_t, count_params_f, magnitude(count_params_t + count_params_f)])
            total_params_train += count_params_t
            total_params_freeze += count_params_f
        table.add_row(['--------', '--------', '-----', '-----'])
        table.add_row(['Total', total_params_train, total_params_freeze,
                       magnitude(total_params_train + total_params_freeze)])

    print(table)
    print(f"Total Trainable Params: {total_params_train + total_params_freeze}")
    return total_params_train + total_params_freeze


