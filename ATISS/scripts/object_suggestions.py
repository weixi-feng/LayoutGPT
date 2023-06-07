# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

#!/usr/bin/env python
import argparse
import logging
import os
import sys

import numpy as np
import torch

import trimesh

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, \
    poll_specific_class, make_network_input, \
    render_scene_from_bbox_params

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Scene, Mesh
from simple_3dviz.window import show
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render


def poll_objects(dataset, current_boxes, scene_id):
    """Show the objects in the current_scene and ask which ones to be
    removed."""
    object_types = np.array(dataset.object_types)
    labels = object_types[current_boxes["class_labels"].argmax(-1)].tolist()
    print(
        "The {} scene you selected contains {}".format(
            scene_id, list(enumerate(labels))
        )
    )
    msg = "Enter the indices of objects to be removed, separated with commas\n"
    ois = [int(oi) for oi in input(msg).split(",") if oi != ""]
    idxs_kept = list(set(range(len(labels))) - set(ois))
    print("You are keeping the following indices {}".format(idxs_kept))

    bbox_bounds = [
        float(ti)
        for ti in input("Enter bbox dims to place an object\n").split(",")
    ]

    return idxs_kept, bbox_bounds


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    bbox_bounds,
    add_start_end=False,
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels)
    )

    box_renderable, box_trimesh = post_process_box(dataset, bbox_bounds)

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, tr_floor + [box_trimesh] + trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render(
        renderables + floor_plan + [box_renderable],
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def post_process_box(dataset, bbox_bounds):
    boxes = dict(
        class_labels=torch.zeros(1, 2, dataset.n_classes),
        sizes=torch.zeros(1, 2, 3),
        angles=torch.zeros(1, 2, 1),
        translations=torch.tensor([
            [bbox_bounds[0], bbox_bounds[2], bbox_bounds[4]],
            [bbox_bounds[1], bbox_bounds[3], bbox_bounds[5]]
        ])[None]
    )
    boxes = dataset.post_process(boxes)

    box_min = boxes["translations"][0, 0].numpy()
    box_max = boxes["translations"][0, 1].numpy()
    centroid = (box_max + box_min)/2
    radii = (box_max - box_min)/2

    box_renderable = Mesh.from_boxes(centroid[None], radii[None], (1.0, 0, 0))
    box_trimesh = trimesh.creation.box(extents=2*radii)
    box_trimesh.apply_translation(centroid)
    box_trimesh.visual.face_colors = [1.0, 0, 0, 1.0]

    return box_renderable, box_trimesh


def sample_in_bbox(class_probs, translation_probs, bbox, trials=1000):
    """Do rejection sampling to sample the class and translation from the given
    probabilities."""
    def in_bbox(bbox, x, y, z):
        return (
            bbox[0] <= x <= bbox[1] and
            bbox[2] <= y <= bbox[3] and
            bbox[4] <= z <= bbox[5]
        )

    def sample_dmll(probs, mu, s):
        i = np.random.choice(len(probs), p=probs)
        u = np.random.rand()
        return np.clip(
            mu[i] + s[i] * (np.log(u) - np.log(1-u)),
            -1,
            1
        )

    # Prepare the probs for sampling (casting to numpy basically)
    class_probs = class_probs.numpy().ravel()
    translation_probs = [
        [
            (p.numpy().ravel(), mu.numpy().ravel(), s.numpy().ravel())
            for (p, mu, s) in lc
        ] for lc in translation_probs
    ]

    # How many trials to do before giving up
    N = trials

    # Sample the class labels
    classes = np.random.choice(len(class_probs), N, p=class_probs)
    for i in range(N):
        if classes[i] >= len(translation_probs):
            continue

        c = classes[i]
        x, y, z = [sample_dmll(*di) for di in translation_probs[c]]
        if in_bbox(bbox, x, y, z):
            return c, (x, y, z)

    raise RuntimeError("Couldn't sample in the bbox")


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Suggest a new object to be added based on a user "
                     "specified region of acceptable positions")
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
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
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
    parser.add_argument(
        "--trials",
        type=int,
        default=100000,
        help="How many trials to do for rejection sampling"
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

        object_indices, bbox_bounds = poll_objects(
            dataset, current_boxes, current_scene.scene_id
        )
        boxes = make_network_input(current_boxes, object_indices)

        # Render the partial
        render_to_folder(
            args,
            "input_{}_{:03}".format(current_scene.scene_id, i),
            dataset,
            objects_dataset,
            tr_floor,
            floor_plan,
            scene,
            boxes,
            bbox_bounds,
            True,
        )

        # Given the current context predict the probability of all class labels
        with torch.no_grad():
            class_probs = network.distribution_classes(
                boxes=boxes, room_mask=room_mask
            )
            translation_probs = [
                network.distribution_translations(boxes, room_mask, c)
                for c in range(len(dataset.object_types))
            ]
        try:
            new_class, (tx, ty, tz) = sample_in_bbox(class_probs,
                                                     translation_probs,
                                                     bbox_bounds, args.trials)
        except RuntimeError:
            continue
        print("Adding {} at location: ({:.4f}, {:.4f}, {:.4f})".format(
            dataset.object_types[new_class], tx, ty, tz
        ))

        bbox_params = network.add_object_with_class_and_translation(
            boxes,
            room_mask,
            int(new_class),
            torch.tensor([tx, ty, tz])[None, None].float()
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
            "suggestion_{}_{:03d}".format(current_scene.scene_id, i)
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
