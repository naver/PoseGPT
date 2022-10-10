# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

def add_histogram(writer, hist, tag, global_step=None, walltime=None):
    """ Add an histogram to tensorboard. """
    NUM = 10000
    hist = NUM * hist
    codebook_size = hist.shape[-1]
    _sum = sum([e * (i + 0.5) for i, e in enumerate(hist)])
    sum_sq = sum([(e * (i + 0.5))**2 for i, e in enumerate(hist)])
    writer.add_histogram_raw(tag=tag,
                             min=0,
                             max=codebook_size - 1,
                             num=NUM,
                             sum=_sum,
                             sum_squares=sum_sq,
                             bucket_limits=list(range(1,codebook_size+1)),
                             bucket_counts=[e for e in hist],
                             global_step=global_step,
                             walltime=walltime)
