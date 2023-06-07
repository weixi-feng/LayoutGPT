# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

from functools import partial
import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .autoregressive_transformer import AutoregressiveTransformer, \
    AutoregressiveTransformerPE, \
    train_on_batch as train_on_batch_simple_autoregressive, \
    validate_on_batch as validate_on_batch_simple_autoregressive

from .hidden_to_output import AutoregressiveDMLL, get_bbox_output
from .feature_extractors import get_feature_extractor


def hidden2output_layer(config, n_classes):
    config_n = config["network"]
    hidden2output_layer = config_n.get("hidden2output_layer")

    if hidden2output_layer == "autoregressive_mlc":
        return AutoregressiveDMLL(
            config_n.get("hidden_dims", 768),
            n_classes,
            config_n.get("n_mixtures", 4),
            get_bbox_output(config_n.get("bbox_output", "autoregressive_mlc")),
            config_n.get("with_extra_fc", False),
        )
    else:
        raise NotImplementedError()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    # weight_decay = config.get("weight_decay", 0.0)
    # Weight decay was set to 0.0 in the paper's experiments. We note that
    # increasing the weight_decay deteriorates performance.
    weight_decay = 0.0

    if optimizer == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()


def build_network(
    input_dims,
    n_classes,
    config,
    weight_file=None,
    device="cpu"):
    network_type = config["network"]["type"]

    if network_type == "autoregressive_transformer":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformer(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    elif network_type == "autoregressive_transformer_pe":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformerPE(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    else:
        raise NotImplementedError()

    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )
    network.to(device)
    return network, train_on_batch, validate_on_batch
