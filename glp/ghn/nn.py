from typing import Mapping
import torch
import torch.nn as nn
import os
#from .mlp import MLP
from .gnn.gatedgnn import GatedGNN
from .decoder import MLPDecoder, ConvDecoder
from .encoder import MLPEncoder, ConvEncoder
from .layers import ShapeEncoder
from ..deepnets1m.ops import NormLayers, PosEnc
from ..deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
from ..deepnets1m.net import named_layered_modules
from ..deepnets1m.graph import Graph, GraphBatch
from ..utils import capacity, default_device
import time



class GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)
    """
    def __init__(self,
                 max_shape,
                 num_classes=10 ,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0,
                 device="cpu"):
        super(GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.hid = hid

        self.device = device

        if layernorm:
            self.ln = nn.LayerNorm(hid*2)

        if hypernet == 'gatedgnn':
            self.gnn = GatedGNN(in_features=hid*2, ve=ve)
        else:
            raise NotImplementedError(hypernet)
        #elif hypernet == 'mlp':
        #    self.gnn = MLP(in_features=hid, hid=(hid, hid))

        self.max_shape = max_shape # [64,64,3,3]

        self.conv_enc = ConvEncoder(self.max_shape,out_features=hid)
        self.linear_enc = MLPEncoder(max_shape[:2],out_features=hid)
        self.bias_enc = nn.Linear(max_shape[0],hid)

        self.predictor = nn.Linear(hid*4,1)

        self.layer_embed = nn.Embedding(len(PRIMITIVES_DEEPNETS1M)+1,hid)

    def forward(self, net, graph):
        param_dict = dict(net.named_parameters())
        features = torch.zeros((graph.n_nodes,self.hid*2), device=self.device)
        features[0,:] = 1
        for ind, (name,param) in enumerate(graph.node_params[1:]):
            if param in net.state_dict().keys():
                weight = param_dict[param][0]
                if weight.ndimension() == 4:
                    in_weight = torch.zeros(self.max_shape,device=self.device)
                    in_weight[:weight.size(0),:weight.size(1),:weight.size(2),:weight.size(3)] = weight
                    features[ind+1,:self.hid] = self.conv_enc(in_weight)
                elif weight.ndimension() == 2:
                    in_weight = torch.zeros(self.max_shape[:2],device=self.device)
                    in_weight[:weight.size(0),:weight.size(1)] = weight
                    features[ind+1,:self.hid] = self.linear_enc(in_weight)
                elif weight.ndimension() == 1:
                    in_weight = torch.zeros(self.max_shape[0],device=self.device)
                    in_weight[:weight.size(0)] = weight
                    features[ind+1,:self.hid] = self.bias_enc(in_weight)
            prim_ind = PRIMITIVES_DEEPNETS1M.index(name) if name in PRIMITIVES_DEEPNETS1M else len(PRIMITIVES_DEEPNETS1M)
            features[ind+1,self.hid:] = self.layer_embed(torch.tensor([prim_ind], device=self.device)).squeeze(0)

        #param_groups, params_map = self._map_net_params(graph, net, self.debug_level > 0)
        #shape_features = self.shape_enc(self.embed(graph.node_feat[:, 0]), params_map, predict_class_layers=True)
        #print(shape_features.shape)

        x = self.gnn(features, graph.edges, graph.node_feat[:,1])
        if self.layernorm:
            x = self.ln(x)
        x_bar = x.sum(0, keepdim = True).repeat(x.size(0),1) 

        out = self.predictor(torch.max(torch.cat([x,x_bar],dim=1),dim=0).values)

        return out


