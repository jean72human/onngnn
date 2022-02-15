import numpy as np
import torch.nn as nn
from .layers import get_activation, MLP


class MLPDecoder(nn.Module):
    def __init__(self,
                 out_shape,
                 in_features=32,
                 hid=(64,)):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.mlp = nn.Linear(in_features,np.prod(out_shape))


    def forward(self, x, max_shape=(0,0)):
        x = self.mlp(x).view(*self.out_shape)
        #if sum(max_shape) > 0:
        #    x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x

class ConvDecoder(nn.Module):
    def __init__(self,
                 out_shape,
                 in_features=32,
                 hid=(128, 256)):
        super(ConvDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.fc = nn.Sequential(nn.Linear(in_features,
                                          hid[0] * np.prod(out_shape[2:])))

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = np.prod(out_shape[:2]) if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1)])

        self.conv = nn.Sequential(*conv)

    def forward(self, x, max_shape=(0,0)):

        N = 1
        x = self.fc(x).view(N, -1, *self.out_shape[2:])  # N,128,11,11
        out_shape = self.out_shape

        x = self.conv(x).view(*out_shape)  # N, out, in, h, w

        return x