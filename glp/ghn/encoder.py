import numpy as np
import torch.nn as nn
from .layers import get_activation, MLP


class MLPEncoder(nn.Module):
    def __init__(self,
                 in_shape,
                 hid=(64,),
                 out_features=32):
        super(MLPEncoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_features = out_features
        self.mlp = nn.Linear(np.prod(in_shape),out_features)


    def forward(self, x, max_shape=(0,0)):
        x = self.mlp(x.flatten())
        #if sum(max_shape) > 0:
        #    x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x

class ConvEncoder(nn.Module):
    def __init__(self,
                 in_shape,
                 hid=(256, 128),
                 out_features=32):
        super(ConvEncoder, self).__init__()

        assert len(hid) > 0, hid
        self.in_shape = in_shape
        self.fc = nn.Sequential(nn.Linear(hid[1] * np.prod(in_shape[2:]),
                                          out_features))

        conv = []
        for j, n_hid in enumerate(hid):
            n_in = np.prod(in_shape[:2]) if j == 0 else hid[j-1]
            conv.extend([nn.Conv2d(n_in, n_hid, 1)])

        self.conv = nn.Sequential(*conv)

    def forward(self, x, max_shape=(0,0)):
        x = self.conv(x.view(-1,*x.shape[2:]).unsqueeze(0)).view(-1)
        x = self.fc(x) 

        return x