# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script that identifies and corrects problematic objects."""
import argparse
import logging
import os
import sys

import numpy as np
import torch

import trimesh

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, render_to_folder, \
    render_scene_from_bbox_params

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Scene
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render


def poll_object(dataset, current_boxes, scene_id):
    """Show the objects in the current_scene and ask which ones to be
    removed."""
    object_types = np.array(dataset.object_types)
    labels = object_types[current_boxes["class_labels"].argmax(-1)].tolist()
    print(
        "The {} scene you selected contains {}".format(
            scene_id, list(enumerate(labels))
        )
    )
    idx = int(input("Enter the index of the object to be moved\n"))
    offset = [float(ti) for ti in input("Enter translation\n").split(",")]
    rotation = float(input("Enter the rotation amount\n"))

    return idx, offset, rotation


def make_network_input(current_boxes, idx, offset, rotation):
    def _prepare(x):
        return torch.from_numpy(np.copy(x[None]).astype(np.float32))

    boxes = dict(
        class_labels=_prepare(current_boxes["class_labels"]),
        translations=_prepare(current_boxes["translations"]),
        sizes=_prepare(current_boxes["sizes"]),
        angles=_prepare(current_boxes["angles"]),
        room_layout=_prepare(current_boxes["room_layout"])
    )
    boxes["translations"][0, idx] += torch.tensor(offset).float()
    boxes["angles"][0, idx] = (boxes["angles"][0, idx] + 1 + rotation) % 2 - 1

    return boxes


def extract_object(boxes, idx):
    n_objects = boxes["class_labels"].shape[1]
    indices = sorted(list(set(range(n_objects)) - set([idx])))
    return dict(
        class_labels=boxes["class_labels"][:, indices],
        translations=boxes["translations"][:, indices],
        sizes=boxes["sizes"][:, indices],
        angles=boxes["angles"][:, indices],
        room_layout=boxes["room_layout"],
        class_labels_tr=boxes["class_labels"][:, idx:idx+1],
        translations_tr=boxes["translations"][:, idx:idx+1],
        angles_tr=boxes["angles"][:, idx:idx+1],
        sizes_tr=boxes["sizes"][:, idx:idx+1],
        lengths=torch.tensor([n_objects-1])
    )


def main(argv):
    parser = argparse.ArgumentParser(
        description="Perform failure cases detection and correction"
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
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
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

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels)
    for i in range(args.n_sequences):
        scene_idx = given_scene_id or np.random.choice(len(dataset))
        current_scene = raw_dataset[scene_idx]
        current_boxes = dataset[scene_idx]
        print("{} / {}: Using the {} floor plan of scene {}".format(
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        # Get a floor plan
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )

        # Choose
        idx, offset, rotation = poll_object(
            dataset, current_boxes, current_scene.scene_id
        )
        boxes = make_network_input(
            current_boxes,
            idx,
            offset,
            rotation
        )

        # Render the failed scene
        render_to_folder(
            args,
            "failure_{}_{:03}".format(current_scene.scene_id, i),
            dataset,
            objects_dataset,
            tr_floor,
            floor_plan,
            scene,
            boxes,
            True
        )

        # Compute object probabilities and pick out the worst
        n_objects = boxes["class_labels"].shape[1]
        probabilities = []
        with torch.no_grad():
            for o in range(n_objects):
                new_boxes = extract_object(boxes, o)
                probabilities.append(
                    -network(new_boxes).reconstruction_loss(
                        new_boxes,
                        lengths=new_boxes["lengths"]
                    )
                )
        worst = np.argmin(probabilities)
        print("The least probable object is", worst)

        # Re-insert that particular object in the room
        new_boxes = extract_object(boxes, worst)
        new_boxes["centroids"] = new_boxes["translations"]
        bbox_params = network.add_object(
            room_mask=room_mask,
            class_label=new_boxes["class_labels_tr"][0, 0].argmax(-1).item(),
            boxes=new_boxes
        )


        # Specify the path of the rendered image
        path_to_image = "{}/{}_{}_{:03d}".format(
            args.output_directory,
            current_scene.scene_id,
            scene_idx,
            i
        )
        # Specify the path to the save the generated scene
        path_to_objs = os.path.join(
            args.output_directory,
            "fixed_{}_{:03d}".format(current_scene.scene_id, i)
        )
        render_scene_from_bbox_params(
            args,
            bbox_params,
            dataset,
            objects_dataset,
            classes,
            floor_plan, 
            tr_floor,
            scene,
            path_to_image,
            path_to_objs
        )


if __name__ == "__main__":
    main(sys.argv[1:])
