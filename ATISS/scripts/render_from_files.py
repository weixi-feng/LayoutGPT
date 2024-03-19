# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import copy
import difflib
import json
import logging
import os
import os.path as op
import pdb
import sys


import numpy as np
from PIL import Image
import pyrr
from tqdm import tqdm
import trimesh

from training_utils import load_config
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded

from simple_3dviz import Scene, Mesh
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.renderables.textured_mesh import TexturedMesh, Material
# from simple_3dviz.utils import render
from utils import render
from simple_3dviz.window import show
from simple_3dviz.io import read_mesh_file

from utils import floor_plan_from_scene, export_scene, get_3d_box, box3d_iou


def scene_init(mesh, up_vector, camera_position, camera_target, background):
    def inner(scene):
        scene.background = background
        scene.up_vector = up_vector
        scene.camera_position = camera_position
        scene.camera_target = camera_target
        scene.light = camera_position
        if mesh is not None:
            scene.add(mesh)
    return inner


def load_room_boxes(prefix, id, stats):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset  = min(data['floor_plan_vertices'][:,0])
    y_offset = min(data['floor_plan_vertices'][:,2])
    room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
    room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    

    return room_length, room_width, x_c, y_c, x_offset, y_offset


def denormalize(predictions, prefix, is_bedroom):
    for v in predictions:
        rl, rw, x_c, y_c, x_offset, y_offset = load_room_boxes(prefix, v['query_id'], None)
        
        if is_bedroom:
            norm = min(rl, rw) / 256.
        else:
            norm = min(rl, rw) / 256.
                
        for _, box in v['object_list']:
            for attr_name, attr_value in box.items():
                if attr_name == 'orientation': 
                    box[attr_name] = (attr_value / 180.) * np.pi
                else:
                    box[attr_name] = attr_value*norm

                if attr_name == 'left':
                    box[attr_name] += (x_offset-x_c)
                if attr_name == 'top':
                    box[attr_name] += (y_offset-y_c)
                    
                if attr_name in ['length', 'width', 'height']:
                    box[attr_name] /= 2.
    return predictions


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    # parser.add_argument(
    #     "path_to_3d_front_dataset_directory",
    #     help="Path to the 3D-FRONT dataset"
    # )
    # parser.add_argument(
    #     "path_to_3d_future_dataset_directory",
    #     help="Path to the 3D-FUTURE dataset"
    # )
    # parser.add_argument(
    #     "path_to_model_info",
    #     help="Path to the 3D-FUTURE model_info.json file"
    # )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "model_output",
        help="Path to model output"
    )
    # parser.add_argument(
    #     "--annotation_file",
    #     default="../config/bedroom_threed_front_splits.csv",
    #     help="Path to the train/test splits file"
    # )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--without_orthographic_projection",
        action="store_true",
        help="Use orthographic projection"
    )
    parser.add_argument(
        "--without_floor_layout",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_door_and_windows",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    # parser.add_argument(
    #     "--with_texture",
    #     action="store_true",
    #     help="Visualize objects with texture"
    # )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=1,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--export_scene",
        action="store_true",
        help="Export scene"
    )
    parser.add_argument(
        "--scene_id",
        nargs="+",
        default=None,
        help="Particular scene to render, e.g. Bedroom-803"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'test', 'val', 'test_regular', 'train_regular', 'val_regular'],
        default='test'
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if args.export_scene:
        args.up_vector = (0.,1.,0.)
        args.camera_position = (2.,2.,2.)
        args.window_size = (int(512), int(512))
        args.background = [1,1,1,1]

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    if not args.without_orthographic_projection:
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-args.room_side, right=args.room_side, 
            bottom=args.room_side, top=-args.room_side, 
            near=0.1, far=6 # 1000
        )
    scene.light = args.camera_position
    behaviours = []

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    config = load_config(args.config_file)
    if '_regular' in args.split:
        config['data']['annotation_file'] = config['data']['annotation_file'].replace("_new.csv", 
                                                                                      "_regular.csv")
        args.split = args.split.split("_")[0]

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=[args.split]
        ),
        split=[args.split]
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    classes = np.array(dataset.class_labels)

    predictions = json.load(open(args.model_output, "r"))
    
    if 'px' in op.basename(args.model_output):
        predictions = denormalize(predictions, config['data']['dataset_directory'], 'bedroom' in args.path_to_pickled_3d_futute_models)
        
    all_ids = [pred['query_id'].split("_")[-1] for pred in predictions]
    predictions = {pred['query_id'].split("_")[-1]:pred for pred in predictions}

    if args.scene_id is not None:
        all_ids = args.scene_id

    print(args.output_directory)
    for s in tqdm(raw_dataset):
        if s.scene_id not in all_ids: continue

        bbox_params = [np.zeros(len(classes)+3+3+1)]
        for obj in predictions[str(s.scene_id)]['object_list']:
            try:
                label_idx = np.where(classes == difflib.get_close_matches(obj[0], classes, cutoff=0.0)[0])[0]
            except:
                continue
            label = np.zeros(len(classes))
            label[label_idx] = 1
            translation = np.asarray([obj[1]['left'], obj[1]['depth'], obj[1]['top']])
            size = np.asarray([obj[1]['length'], obj[1]['height'], obj[1]['width']])
            angle = np.asarray([obj[1]['orientation']])
            param = np.concatenate([label, translation, size, angle])
            bbox_params.append(param)
        bbox_params.append(np.zeros(len(classes)+3+3+1))
        bbox_params_t = np.stack(bbox_params)[None, ...]
        
        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        
        if not args.without_floor_layout:
            # Get a floor plan
            floor_plan, tr_floor, _ = floor_plan_from_scene(
                s, args.path_to_floor_plan_textures, without_room_mask=True
            )
            renderables += floor_plan
            trimesh_meshes += tr_floor

        if args.without_screen:
            path_to_image = "{}/ours_{}.png".format(args.output_directory, s.scene_id)
            behaviours += [
                LightToCamera(), # NOTE
                SaveFrames(path_to_image+"{:03d}.png", 1)
                ]
            
            if args.with_rotating_camera:
                behaviours += [
                    CameraTrajectory(
                        Circle(
                            [0, args.camera_position[1], 0],
                            args.camera_position,
                            args.up_vector
                        ),
                        speed=1/360
                    ),
                    SaveGif(path_to_image+".gif", 1)
                ]

            scene.up_vector = args.up_vector
            scene.camera_target = args.camera_target
            scene.camera_position = args.camera_position
            render(
                scene,
                renderables,
                color=None,
                mode='shading',
                frame_path=path_to_image
            )
        else:
            show(
                renderables,
                behaviours=behaviours+[SnapshotOnKey()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                light=args.camera_position,
                up_vector=args.up_vector,
                background=args.background,
            )

        if args.export_scene:
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "{}_{}".format(args.split, s.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)

if __name__ == "__main__":
    main(sys.argv[1:])
