# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FrozenBatchNorm2d(nn.Module):
    """A BatchNorm2d wrapper for Pytorch's BatchNorm2d where the batch
    statictis are fixed.
    """
    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.register_parameter("weight", Parameter(torch.ones(num_features)))
        self.register_parameter("bias", Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)

    @classmethod
    def from_batch_norm(cls, bn):
        fbn = cls(bn.num_features)
        # Update the weight and biases based on the corresponding weights and
        # biases of the pre-trained bn layer
        with torch.no_grad():
            fbn.weight[...] = bn.weight
            fbn.bias[...] = bn.bias
            fbn.running_mean[...] = bn.running_mean
            fbn.running_var[...] = bn.running_var + bn.eps
        return fbn

    @staticmethod
    def _getattr_nested(m, module_names):
        if len(module_names) == 1:
            return getattr(m, module_names[0])
        else:
            return FrozenBatchNorm2d._getattr_nested(
                getattr(m, module_names[0]), module_names[1:]
            )

    @staticmethod
    def freeze(m):
        for (name, layer) in m.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                nest = name.split(".")
                if len(nest) == 1:
                    setattr(m, name, FrozenBatchNorm2d.from_batch_norm(layer))
                else:
                    setattr(
                        FrozenBatchNorm2d._getattr_nested(m, nest[:-1]),
                        nest[-1],
                        FrozenBatchNorm2d.from_batch_norm(layer)
                    )

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


def freeze_network(network, freeze=False):
    if freeze:
        for p in network.parameters():
            p.requires_grad = False
    return network
