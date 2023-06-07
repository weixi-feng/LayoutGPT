# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
from torch import nn
from torchvision import models

from .frozen_batchnorm import FrozenBatchNorm2d


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks.
    """
    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, X):
        return self._feature_extractor(X)


class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs.
    """
    def __init__(self, freeze_bn, input_channels, feature_size):
        super(ResNet18, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.resnet18(pretrained=False)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class AlexNet(BaseFeatureExtractor):
    def __init__(self, input_channels, feature_size):
        super(AlexNet, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.alexnet(pretrained=False)
        self._feature_extractor.features[0] = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(11, 11),
            stride=(4, 4),
            padding=(2, 2),
        )

        self._fc = nn.Linear(9216, self._feature_size)

    def forward(self, X):
        X = self._feature_extractor.features(X)
        X = self._feature_extractor.avgpool(X)
        X = self._fc(X.view(X.shape[0], -1))

        return X


def get_feature_extractor(
    name,
    freeze_bn=False,
    input_channels=1,
    feature_size=128
):
    """Based on the name return the appropriate feature extractor."""
    return {
        "resnet18": ResNet18(
            freeze_bn=freeze_bn,
            input_channels=input_channels,
            feature_size=feature_size
        ),
        "alexnet": AlexNet(input_channels, feature_size=feature_size)
    }[name]
