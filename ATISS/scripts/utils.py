# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import os

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44

import trimesh

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.io import read_mesh_file
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects

from scipy.spatial import ConvexHull


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid
    # Return a simple-3dviz renderable
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,
    path_to_floor_plan_textures,
    without_room_mask=False
):
    if not without_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_mask = None
    # Also get a renderable for the floor plan
    floor, tr_floor = get_floor_plan(
        scene,
        [
            os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)
        ]
    )
    return [floor], [tr_floor], room_mask


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


def get_textured_objects_in_scene(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes:
        model_path = furniture.raw_model_path
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset
        # raw_mesh = TexturedMesh.from_file(model_path)
        try:
            raw_mesh = TexturedMesh.from_file(model_path)
        except:
            try:
                texture_path = furniture.texture_image_path
                mesh_info = read_mesh_file(model_path)
                vertices = mesh_info.vertices
                normals = mesh_info.normals
                uv = mesh_info.uv
                material = Material.with_texture_image(texture_path)
                raw_mesh = TexturedMesh(vertices,normals,uv,material)
            except:
                print("Failed loading texture info.")
                raw_mesh = Mesh.from_file(model_path)
        
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
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
    return renderables


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes, names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            m,
            return_texture=True
        )

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:-1].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    add_start_end=False
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
    trimesh_meshes += tr_floor

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render_simple_3dviz(
        renderables + floor_plan,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def render_scene_from_bbox_params(
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
):
    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t, objects_dataset, classes
    )
    renderables += floor_plan
    trimesh_meshes += tr_floor

    # Do the rendering
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image+".png", 1)
    ]
    render_simple_3dviz(
        renderables,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
        export_scene(path_to_objs, trimesh_meshes)


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = np.maximum(corners_3d[0,:] + center[0], 0)
    corners_3d[1,:] = np.maximum(corners_3d[1,:] + center[1], 0)
    corners_3d[2,:] = np.maximum(corners_3d[2,:] + center[2], 0)
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    ''' 
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    # iou = inter_vol / (vol1 + vol2 - inter_vol)
    iou = inter_vol / min(vol1, vol2)
    return iou, iou_2d