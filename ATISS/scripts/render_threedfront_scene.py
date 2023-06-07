# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import logging
import os
import pdb
import sys

import numpy as np
from PIL import Image
import pyrr
import trimesh

from scene_synthesis.datasets.threed_front import ThreedFront

from simple_3dviz import Scene
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.utils import render
from simple_3dviz.window import show
from simple_3dviz.renderables.textured_mesh import Material
from simple_3dviz.io import read_mesh_file
from simple_3dviz import Mesh

from utils import floor_plan_from_scene, export_scene


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


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "scene_id",
        nargs="+",
        help="The scene id of the scene to be visualized"
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
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
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="2.0,2.0,2.0",
        help="Camer position in the scene"
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
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--with_orthographic_projection",
        action="store_true",
        help="Use orthographic projection"
    )
    parser.add_argument(
        "--with_floor_layout",
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
    parser.add_argument(
        "--with_texture",
        action="store_true",
        help="Visualize objects with texture"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    if args.with_orthographic_projection:
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-args.room_side, right=args.room_side, bottom=args.room_side, top=-args.room_side, near=0.1, far=1000
        )
    scene.light = args.camera_position
    behaviours = []

    d = ThreedFront.from_dataset_directory(
        args.path_to_3d_front_dataset_directory,
        args.path_to_model_info,
        args.path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s
    )
    print("Loading dataset with {} rooms".format(len(d)))

    for s in d.scenes:
        if s.scene_id in args.scene_id:
            for b in s.bboxes:
                print(b.model_jid, b.label)
            print(s.furniture_in_room, s.scene_id, s.json_path)
            renderables = s.furniture_renderables(
                with_floor_plan_offset=True, with_texture=args.with_texture
            )
            trimesh_meshes = []
            for furniture in s.bboxes:
                # Load the furniture and scale it as it is given in the dataset
                # raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                try:
                    raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                except:
                    try:
                        texture_path = furniture.texture_image_path
                        mesh_info = read_mesh_file(furniture.raw_model_path)
                        vertices = mesh_info.vertices
                        normals = mesh_info.normals
                        uv = mesh_info.uv
                        material = Material.with_texture_image(texture_path)
                        raw_mesh = TexturedMesh(vertices,normals,uv,material)
                    except:
                        print("Failed loading texture info.")
                        raw_mesh = Mesh.from_file(furniture.raw_model_path)
                        
                raw_mesh.scale(furniture.scale)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                tr_mesh.vertices *= furniture.scale
                theta = furniture.z_angle
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.
                tr_mesh.vertices[...] = \
                    tr_mesh.vertices.dot(R) + furniture.position
                tr_mesh.vertices[...] = tr_mesh.vertices - s.centroid
                trimesh_meshes.append(tr_mesh)

            if args.with_floor_layout:
                # Get a floor plan
                floor_plan, tr_floor, _ = floor_plan_from_scene(
                    s, args.path_to_floor_plan_textures, without_room_mask=True
                )
                renderables += floor_plan
                trimesh_meshes += tr_floor

            if args.with_walls:
                for ei in s.extras:
                    if "WallInner" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if args.with_door_and_windows:
                for ei in s.extras:
                    if "Window" in ei.model_type or "Door" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if args.without_screen:
                path_to_image = "{}/{}_".format(args.output_directory, s.uid)
                behaviours += [SaveFrames(path_to_image+"{:03d}.png", 1)]
                render(
                    renderables,
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    up_vector=args.up_vector,
                    background=args.background,
                    behaviours=behaviours,
                    n_frames=1,
                    scene=scene
                )
                scene.clear()
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
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "train_{}".format(args.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])
