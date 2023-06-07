# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for estimating the KL-divergence between the object categories
of real and generated scenes."""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from tqdm import tqdm

from training_utils import load_config

from scene_synthesis.datasets import get_dataset_raw_and_encoded
from scene_synthesis.datasets import filter_function
from scene_synthesis.networks import build_network


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the KL-divergence between the object category "
                     "distributions of real and synthesized scenes")
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--path_to_bounds",
        default=None,
        help="Path to the dataset's bounds"
    )
    parser.add_argument(
        "--output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--n_synthesized_scenes",
        default=1000,
        type=int,
        help="Number of scenes to be synthesized"
    )
    parser.add_argument(
        "--splits",
        choices=[
            "training",
            "validation"
        ],
        default="training",
        help="Split to evaluate"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    # Load the training dataset
    path_to_bounds = args.path_to_bounds

    config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_eval"
    dataset, ground_truth_scenes = get_dataset_raw_and_encoded(
        config["data"],
        filter_function(
            config["data"],
            split=config[args.splits].get("splits", ["test"]),
            without_lamps=args.without_lamps
        ),
        path_to_bounds=args.path_to_bounds,
        split=config[args.splits].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    # Using the pre-trained model create the synthesized dataset
    network, _, validate_on_batch = build_network(
        ground_truth_scenes.feature_size, ground_truth_scenes.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Generate some rooms with the pre-trained model
    synthesized_scenes = []
    for i in tqdm(range(args.n_synthesized_scenes)):
        scene_idx = np.random.choice(len(dataset))
        scene = dataset[scene_idx]
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        ).to(device)
        bbox_params = network.generate_boxes(
            room_mask=room_mask, device=device
        )
        boxes = dataset.post_process(bbox_params)
        synthesized_scenes.append({
            k: v[0].cpu().numpy()
            for k, v in boxes.items()
        })
    print(dataset.class_labels)

    # Firstly compute the frequencies of the class labels
    gt_class_labels = sum([
        d["class_labels"].sum(0)
        for d in ground_truth_scenes
    ]) / sum([
        d["class_labels"].shape[0]
        for d in ground_truth_scenes
    ])
    syn_class_labels = sum([
        d["class_labels"][1:-1].sum(0)
        for d in synthesized_scenes
    ]) / sum([
        d["class_labels"].shape[0]-2
        for d in synthesized_scenes
    ])
    assert 0.9999 <= gt_class_labels.sum() <= 1.0001
    assert 0.9999 <= syn_class_labels.sum() <= 1.0001
    stats = {}
    stats["class_labels"] = categorical_kl(gt_class_labels, syn_class_labels)
    print(stats)
    path_to_stats = os.path.join(
        args.output_directory, "{}_stats.npz".format(args.splits)
    )

    classes = np.array(dataset.class_labels)
    for c, gt_cp, syn_cp in zip(classes, gt_class_labels, syn_class_labels):
        print("{}: target: {} / synth: {}".format(c, gt_cp, syn_cp))

    gt_cooccurrences = np.zeros((len(classes)-2, len(classes)-2))
    syn_cooccurrences = np.zeros((len(classes)-2, len(classes)-2))
    for gt_scene, syn_scene in zip(ground_truth_scenes, synthesized_scenes):
        gt_classes = gt_scene["class_labels"].argmax(axis=-1)
        syn_classes = syn_scene["class_labels"][1:-1].argmax(axis=-1)

        for ii in range(len(gt_classes)):
            r = gt_classes[ii]
            for jj in range(ii+1, len(gt_classes)):
                c = gt_classes[jj]
                gt_cooccurrences[r, c] += 1

        for ii in range(len(syn_classes)):
            r = syn_classes[ii]
            for jj in range(ii+1, len(syn_classes)):
                c = syn_classes[jj]
                syn_cooccurrences[r, c] += 1

    print("Saving stats at {}".format(path_to_stats))
    np.savez(
        path_to_stats,
        stats=stats,
        classes=classes,
        gt_class_labels=gt_class_labels,
        syn_class_labels=syn_class_labels,
        gt_cooccurrences=gt_cooccurrences,
        syn_cooccurrences=syn_cooccurrences
    )


if __name__ == "__main__":
    main(sys.argv[1:])
