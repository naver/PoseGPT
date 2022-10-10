# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
from einops import rearrange


# Kernel size .
# Padding to keep the same dimensions.
# Strides, max pooling, average pooling?

class Masked_conv(nn.Module):
    def __init__(self, in_chan, out_chan, masked=True, pool_size=2, pool_type='max'):
        super().__init__()
        assert pool_type in ['max', 'avg']
        self.masked = masked
        self.conv = nn.Conv1d(in_channels=in_chan,
                              out_channels=out_chan,
                              kernel_size=2 if masked else 3,
                              stride=1,
                              padding=(0,) if masked else (1,),
                              padding_mode='zeros')
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
        else:
            print("does not work with the way you handled mask, some work to do")
            import pdb; pdb.set_trace()
            self.pool = nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x, mask=None):
        if self.masked:
            x = torch.cat([torch.zeros_like(x[:, 0, :])[:, None, :], x], dim=1)
        x = x.permute((0,2,1))
        x = self.conv(x)

        if mask is not None:
            # mask dependant down-sampling. A bit hacky but what ever?
            maxval = x.abs().max()
            x = x - (~mask.unsqueeze(1)).int() * 10 * maxval
            x = self.pool(x)
            x = x + (x < - 5 * maxval).int() * 10 * maxval
            if mask.shape[1] % 2:
                mask = torch.cat([mask, mask[:, -1][:, None]], dim=1)
            mask = rearrange(mask, 'b (t t2)-> b t t2', t2=2)[:, :, 0]
        return x.permute((0,2,1)), mask[:, :x.shape[2]]

class Masked_up_conv(nn.Module):
    def __init__(self, in_chan, out_chan):
        # NOTE: We are not explicitely performing masking here, unlike in the convolution,
        # because with a combination of kernel size 3 and stride 2 it's already valid.
        # Not true in general, if you change the kernel size or the stride though.
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_chan,
                                       out_channels=out_chan,
                                       kernel_size=3,
                                       stride=2,
                                       padding=(0,),
                                       padding_mode='zeros')

    def forward(self, x, mask=None):
        x = x.permute((0,2,1))
        y = self.conv(x)
        y = y.permute((0,2,1))[:, 1:, :]
        # Upsample the mask by simply duplicating the values)
        mask = torch.stack([mask, mask], dim=2).reshape(mask.shape[0], -1)
        return y, mask


if __name__ == '__main__':
    x = torch.ones((32, 64, 384))

    conv = Masked_conv(384, 384, masked=False)
    mconv = Masked_conv(384, 384, masked=True)

    y = conv(x)
    yy = mconv(x)
    uconv = Masked_up_conv(384, 384)
    #y == uconv(dconv(x))

    i = 5
    func = lambda x: uconv(dconv(x))[:, i, :].sum()
    y = func(x)

    test = torch.autograd.functional.jacobian(func, x).sum((0,2))


