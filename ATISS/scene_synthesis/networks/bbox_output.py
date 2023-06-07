# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch

from ..losses import cross_entropy_loss, dmll
from ..stats_logger import StatsLogger


class BBoxOutput(object):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes = sizes
        self.translations = translations
        self.angles = angles
        self.class_labels = class_labels

    def __len__(self):
        return len(self.members)

    @property
    def members(self):
        return (self.sizes, self.translations, self.angles, self.class_labels)

    @property
    def n_classes(self):
        return self.class_labels.shape[-1]

    @property
    def device(self):
        return self.class_labels.device

    @staticmethod
    def extract_bbox_params_from_tensor(t):
        if isinstance(t, dict):
            class_labels = t["class_labels_tr"]
            translations = t["translations_tr"]
            sizes = t["sizes_tr"]
            angles = t["angles_tr"]
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]

        return class_labels, translations, sizes, angles

    @property
    def feature_dims(self):
        raise NotImplementedError()

    def get_losses(self, X_target):
        raise NotImplementedError()

    def reconstruction_loss(self, sample_params):
        raise NotImplementedError()


class AutoregressiveBBoxOutput(BBoxOutput):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes_x, self.sizes_y, self.sizes_z = sizes
        self.translations_x, self.translations_y, self.translations_z = \
            translations
        self.class_labels = class_labels
        self.angles = angles

    @property
    def members(self):
        return (
            self.sizes_x, self.sizes_y, self.sizes_z,
            self.translations_x, self.translations_y, self.translations_z,
            self.angles, self.class_labels
        )

    @property
    def feature_dims(self):
        return self.n_classes + 3 + 3 + 1

    def _targets_from_tensor(self, X_target):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_bbox_params_from_tensor(X_target)
        target = {}
        target["labels"] = target_bbox_params[0]
        target["translations_x"] = target_bbox_params[1][:, :, 0:1]
        target["translations_y"] = target_bbox_params[1][:, :, 1:2]
        target["translations_z"] = target_bbox_params[1][:, :, 2:3]
        target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
        target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
        target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
        target["angles"] = target_bbox_params[3]

        return target

    def get_losses(self, X_target):
        target = self._targets_from_tensor(X_target)

        assert torch.sum(target["labels"][..., -2]).item() == 0

        # For the class labels compute the cross entropy loss between the
        # target and the predicted labels
        label_loss = cross_entropy_loss(self.class_labels, target["labels"])

        # For the translations, sizes and angles compute the discretized
        # logistic mixture likelihood as described in 
        # PIXELCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and
        # Other Modifications, by Salimans et al.
        translation_loss = dmll(self.translations_x, target["translations_x"])
        translation_loss += dmll(self.translations_y, target["translations_y"])
        translation_loss += dmll(self.translations_z, target["translations_z"])
        size_loss = dmll(self.sizes_x, target["sizes_x"])
        size_loss += dmll(self.sizes_y, target["sizes_y"])
        size_loss += dmll(self.sizes_z, target["sizes_z"])
        angle_loss = dmll(self.angles, target["angles"])

        return label_loss, translation_loss, size_loss, angle_loss

    def reconstruction_loss(self, X_target, lengths):
        # Compute the losses
        label_loss, translation_loss, size_loss, angle_loss = \
            self.get_losses(X_target)

        label_loss = label_loss.mean()
        translation_loss = translation_loss.mean()
        size_loss = size_loss.mean()
        angle_loss = angle_loss.mean()

        StatsLogger.instance()["losses.size"].value = size_loss.item()
        StatsLogger.instance()["losses.translation"].value = \
            translation_loss.item()
        StatsLogger.instance()["losses.angle"].value = angle_loss.item()
        StatsLogger.instance()["losses.label"].value = label_loss.item()

        return label_loss + translation_loss + size_loss + angle_loss
