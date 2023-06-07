# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used to train a ATISS."""
import argparse
import logging
import os
import pdb
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader

from training_utils import id_generator, save_experiment_params, load_config

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB


def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}"
    ).format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train"]) # NOTE
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train"]) # NOTE
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"]
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["val"]) # NOTE
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["val"]) # NOTE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("Training set has {} bounds".format(train_dataset.bounds))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        config, args.weight_file, device=device
    )
    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], network.parameters())
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)

    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "autoregressive_transformer"
            ),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        network.train()
        for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to device
            for k, v in sample.items():
                sample[k] = v.to(device)
            batch_loss = train_on_batch(network, optimizer, sample, config)
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
