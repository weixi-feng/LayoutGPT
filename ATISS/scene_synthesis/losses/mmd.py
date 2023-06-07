# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch


def expected_gaussian_kernel(X, Y, gamma=1.0):
    """Compute the average of k(X, Y) where k is the gaussian rbf kernel with
    γ=`gamma`.

    In particular, compute

        k(X, Y) = ||exp(-γ ||Χ-Υ||_2^2)||_1 / N / M
                = ||exp(-γ (||Χ||^2 + ||Υ||^2 - 2Χ Υ^Τ))||_1 / N / M

    Arguments
    ---------
        X: tensor (N, D) Features sampled from the first distribution
        Y: tensor (M, D) Features sampled from the second distribution
        gamma: The γ parameter for the rbf kernel.
    """
    N, D = X.shape
    M, _ = Y.shape

    X2 = torch.einsum("nd,nd->n", X, X)
    # minor optimization for the case that Y = X
    if Y is X:
        Y2 = X2
    else:
        Y2 = torch.einsum("md,md->m", Y, Y)
    XY = torch.einsum("nd,md->nm", X, Y)

    # prepare them for addition by making sure
    # X2.shape == (N, 1) and Y2.shape == (1, M)
    X2 = X2.view(N, 1)
    Y2 = Y2.view(1, M)

    return torch.exp(-gamma * (X2 + Y2 - 2*XY)).mean()


def mmd(X, Y, gamma=1.0):
    """Compute the mmd loss with gaussian rbf kernel with γ=`gamma`.

    This function computes

        MMD(X, Y) ~= E_p,p[k(x, x')] + E_q,q[k(y, y')] - 2 E_p,q[k(x, y)]

    Arguments
    ---------
        X: tensor (N, D) Features sampled from the first distribution
        Y: tensor (M, D) Features sampled from the second distribution
        gamma: The γ parameter for the rbf kernel.
    """
    Exx = expected_gaussian_kernel(X, X, gamma)
    Eyy = expected_gaussian_kernel(Y, Y, gamma)
    Exy = expected_gaussian_kernel(X, Y, gamma)

    return Exx + Eyy - 2 * Exy
