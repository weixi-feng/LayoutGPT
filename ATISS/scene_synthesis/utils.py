# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import pdb
import numpy as np
from PIL import Image
import trimesh

from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.io import read_mesh_file
from simple_3dviz import Mesh

def get_textured_objects(bbox_params_t, objects_dataset, classes):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    for j in range(1, bbox_params_t.shape[1]-1):
        query_size = bbox_params_t[0, j, -4:-1]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )

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

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        tr_mesh.visual.material.image = Image.open(
            furniture.texture_image_path
        )
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )

    return floor, tr_floor
