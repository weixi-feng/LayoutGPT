# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn

from .bbox_output import AutoregressiveBBoxOutput
from .base import FixedPositionalEncoding, sample_from_dmll


class Hidden2Output(nn.Module):
    def __init__(self, hidden_size, n_classes, with_extra_fc=False):
        super().__init__()
        self.with_extra_fc = with_extra_fc
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        ]
        self.hidden2output = nn.Sequential(*mlp_layers)

    def apply_linear_layers(self, x):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        class_labels = self.class_layer(x)
        translations = (
            self.centroid_layer_x(x),
            self.centroid_layer_y(x),
            self.centroid_layer_z(x)
        )
        sizes = (
            self.size_layer_x(x),
            self.size_layer_y(x),
            self.size_layer_z(x)
        )
        angles = self.angle_layer(x)
        return class_labels, translations, sizes, angles

    def forward(self, x, sample_params=None):
        raise NotImplementedError()


class AutoregressiveDMLL(Hidden2Output):
    def __init__(
        self,
        hidden_size,
        n_classes,
        n_mixtures,
        bbox_output,
        with_extra_fc=False
    ):
        super().__init__(hidden_size, n_classes, with_extra_fc)

        if not isinstance(n_mixtures, list):
            n_mixtures = [n_mixtures]*7

        self.class_layer = nn.Linear(hidden_size, n_classes)

        self.fc_class_labels = nn.Linear(n_classes, 64)
        # Positional embedding for the target translation
        self.pe_trans_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_z = FixedPositionalEncoding(proj_dims=64)
        # Positional embedding for the target angle
        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        c_hidden_size = hidden_size + 64
        self.centroid_layer_x = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[0]*3
        )
        self.centroid_layer_y = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[1]*3
        )
        self.centroid_layer_z = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[2]*3
        )
        c_hidden_size = c_hidden_size + 64*3
        self.angle_layer = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[6]*3
        )
        c_hidden_size = c_hidden_size + 64
        self.size_layer_x = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[3]*3
        )
        self.size_layer_y = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[4]*3
        )
        self.size_layer_z = AutoregressiveDMLL._mlp(
            c_hidden_size, n_mixtures[5]*3
        )

        self.bbox_output = bbox_output

    @staticmethod
    def _mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _extract_properties_from_target(sample_params):
        class_labels = sample_params["class_labels_tr"].float()
        translations = sample_params["translations_tr"].float()
        sizes = sample_params["sizes_tr"].float()
        angles = sample_params["angles_tr"].float()
        return class_labels, translations, sizes, angles

    @staticmethod
    def get_dmll_params(pred):
        assert len(pred.shape) == 2

        N = pred.size(0)
        nr_mix = pred.size(1) // 3

        probs = torch.softmax(pred[:, :nr_mix], dim=-1)
        means = pred[:, nr_mix:2 * nr_mix]
        scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

        return probs, means, scales

    def get_translations_dmll_params(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        translations_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        translations_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        dmll_params = {}
        p = AutoregressiveDMLL.get_dmll_params(translations_x)
        dmll_params["translations_x_probs"] = p[0]
        dmll_params["translations_x_means"] = p[1]
        dmll_params["translations_x_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_y)
        dmll_params["translations_y_probs"] = p[0]
        dmll_params["translations_y_means"] = p[1]
        dmll_params["translations_y_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_z)
        dmll_params["translations_z_probs"] = p[0]
        dmll_params["translations_z_means"] = p[1]
        dmll_params["translations_z_scales"] = p[2]

        return dmll_params

    def sample_class_labels(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape
        C = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(B*L, C)
        sampled_classes = torch.multinomial(class_probs, 1).view(B, L)
        return torch.eye(C, device=x.device)[sampled_classes]

    def sample_translations(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf)
        translations_y = self.centroid_layer_y(cf)
        translations_z = self.centroid_layer_z(cf)

        t_x = sample_from_dmll(translations_x.reshape(B*L, -1))
        t_y = sample_from_dmll(translations_y.reshape(B*L, -1))
        t_z = sample_from_dmll(translations_z.reshape(B*L, -1))
        return torch.cat([t_x, t_y, t_z], dim=-1).view(B, L, 3)

    def sample_angles(self, x, class_labels, translations):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)
        return sample_from_dmll(angles.reshape(B*L, -1)).view(B, L, 1)

    def sample_sizes(self, x, class_labels, translations, angles):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        a = self.pe_angle_z(angles)
        sf = torch.cat([tf, a], dim=-1)

        sizes_x = self.size_layer_x(sf)
        sizes_y = self.size_layer_y(sf)
        sizes_z = self.size_layer_z(sf)

        s_x = sample_from_dmll(sizes_x.reshape(B*L, -1))
        s_y = sample_from_dmll(sizes_y.reshape(B*L, -1))
        s_z = sample_from_dmll(sizes_z.reshape(B*L, -1))
        return torch.cat([s_x, s_y, s_z], dim=-1).view(B, L, 3)

    def pred_class_probs(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        b, l, _ = class_labels.shape
        c = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(b*l, c)

        return class_probs

    def pred_dmll_params_translation(self, x, class_labels):
        def dmll_params_from_pred(pred):
            assert len(pred.shape) == 2

            N = pred.size(0)
            nr_mix = pred.size(1) // 3

            probs = torch.softmax(pred[:, :nr_mix], dim=-1)
            means = pred[:, nr_mix:2 * nr_mix]
            scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix])
            scales = scales + 1.0001

            return probs, means, scales

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        t_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        t_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        t_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        return dmll_params_from_pred(t_x), dmll_params_from_pred(t_y),\
            dmll_params_from_pred(t_z)

    def forward(self, x, sample_params):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        # Extract the target properties from sample_params and embed them into
        # a higher dimensional space.
        target_properties = \
            AutoregressiveDMLL._extract_properties_from_target(
                sample_params
            )

        class_labels = target_properties[0]
        translations = target_properties[1]
        angles = target_properties[3]

        c = self.fc_class_labels(class_labels)

        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])

        a = self.pe_angle_z(angles)
        class_labels = self.class_layer(x)

        cf = torch.cat([x, c], dim=-1)
        # Using the true class label we now want to predict the translations
        translations = (
            self.centroid_layer_x(cf),
            self.centroid_layer_y(cf),
            self.centroid_layer_z(cf)
        )
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)
        sf = torch.cat([tf, a], dim=-1)
        sizes = (
            self.size_layer_x(sf),
            self.size_layer_y(sf),
            self.size_layer_z(sf)
        )

        return self.bbox_output(sizes, translations, angles, class_labels)


def get_bbox_output(bbox_type):
    return {
        "autoregressive_mlc": AutoregressiveBBoxOutput
    }[bbox_type]
