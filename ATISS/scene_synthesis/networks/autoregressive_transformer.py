# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask

from .base import FixedPositionalEncoding
from ..stats_logger import StatsLogger


class BaseAutoregressiveTransformer(nn.Module):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__()
        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get(
                "feed_forward_dimensions", 3072
            ),
            attention_type="full",
            activation="gelu"
        ).get()

        self.register_parameter(
            "start_token_embedding",
            nn.Parameter(torch.randn(1, 512))
        )

        # TODO: Add the projection dimensions for the room features in the
        # config!!!
        self.feature_extractor = feature_extractor
        self.fc_room_f = nn.Linear(
            self.feature_extractor.feature_size, 512
        )

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3 to account for the masked bins for the size, position and
        # angle properties
        self.input_dims = input_dims
        self.n_classes = self.input_dims - 3 - 3 - 1
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)

        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(512, hidden_dims)
        self.hidden2output = hidden2output

    def start_symbol(self, device="cpu"):
        start_class = torch.zeros(1, 1, self.n_classes, device=device)
        start_class[0, 0, -2] = 1
        return {
            "class_labels": start_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device)
        }

        return boxes

    def end_symbol(self, device="cpu"):
        end_class = torch.zeros(1, 1, self.n_classes, device=device)
        end_class[0, 0, -1] = 1
        return {
            "class_labels": end_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device)
        }

    def start_symbol_features(self, B, room_mask):
        room_layout_f = self.fc_room_f(self.feature_extractor(room_mask))
        return room_layout_f[:, None, :]

    def forward(self, sample_params):
        raise NotImplementedError()

    def autoregressive_decode(self, boxes, room_mask):
        raise NotImplementedError()

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        raise NotImplementedError()


class AutoregressiveTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 512))
        )

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, _, _ = class_labels.shape

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, _, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_y(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_z(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_y(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_z(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    def autoregressive_decode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        # Sample the class label for the next bbbox
        class_labels = self.hidden2output.sample_class_labels(F)
        # Sample the translations
        translations = self.hidden2output.sample_translations(F, class_labels)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_labels, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_labels, translations, angles
        )

        return {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        boxes = self.start_symbol(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu")
        }

    def autoregressive_decode_with_class_label(
        self, boxes, room_mask, class_label
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the translations conditioned on the query_class_label
        translations = self.hidden2output.sample_translations(F, class_label)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_label, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translations, angles
        )

        return {
            "class_labels": class_label,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object(self, room_mask, class_label, boxes=None, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label(
            boxes=boxes,
            room_mask=room_mask,
            class_label=class_label
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def complete_scene(
        self,
        boxes,
        room_mask,
        max_boxes=100,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    def autoregressive_decode_with_class_label_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translation, angles
        )

        return {
            "class_labels": class_label,
            "translations": translation,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object_with_class_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)


        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes,
            class_label=class_label,
            translation=translation,
            room_mask=room_mask
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        room_mask, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )


class AutoregressiveTransformerPE(AutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 512))
        )

        # Positional embedding for the ordering
        max_seq_length = 32
        self.register_parameter(
            "positional_embedding",
            nn.Parameter(torch.randn(max_seq_length, 32))
        )

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=60)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=60)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=60)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=60)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=60)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=60)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3 to account for the masked bins for the size, position and
        # angle properties
        self.input_dims = input_dims
        self.n_classes = self.input_dims - 3 - 3 - 1
        self.fc_class = nn.Linear(self.n_classes, 60, bias=False)

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, L, _ = class_labels.shape

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        pe = self.positional_embedding[None, :L].expand(B, -1, -1)
        X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, L, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_y(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_z(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_y(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_z(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            pe = self.positional_embedding[None, 1:L].expand(B, -1, -1)
            X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F


def train_on_batch(model, optimizer, sample_params, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    return loss.item()
