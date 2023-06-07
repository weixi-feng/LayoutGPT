# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import math
import torch
import torch.nn as nn
torch.pi = math.pi

class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims//2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1)
        self.sigma = 2 * torch.pi * self.sigma

    def forward(self, x):
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1)


def sample_from_dmll(pred, num_classes=256):
    """Sample from mixture of logistics.

    Arguments
    ---------
        pred: NxC where C is 3*number of logistics
    """
    assert len(pred.shape) == 2

    N = pred.size(0)
    nr_mix = pred.size(1) // 3

    probs = torch.softmax(pred[:, :nr_mix], dim=-1)
    means = pred[:, nr_mix:2 * nr_mix]
    scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

    indices = torch.multinomial(probs, 1).squeeze()
    batch_indices = torch.arange(N, device=probs.device)
    mu = means[batch_indices, indices]
    s = scales[batch_indices, indices]
    u = torch.rand(N, device=probs.device)
    preds = mu + s*(torch.log(u) - torch.log(1-u))

    return torch.clamp(preds, min=-1, max=1)[:, None]


def optimizer_factory(config, parameters):
    """Based on the input arguments create a suitable optimizer object."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)

    if optimizer == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise NotImplementedError()
