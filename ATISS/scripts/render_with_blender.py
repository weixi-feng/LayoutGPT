from glob import glob
import pdb
import bpy
import os
import argparse

scene = bpy.context.scene

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="directory to a scene folder with object and material files")
parser.add_argument("--output_dir", type=str, default="./blender_visulization.png", help="directory to save the rendered image")
parser.add_argument("--camera_position", nargs="+", type=float, default=[5,5,5], help="camera position in XYZ order")

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)
    bpy.data.lights.remove(bpy.data.lights["Light"], do_unlink=True)

    # add a new light
    bpy.ops.object.light_add(type="SUN")
    bpy.data.lights["Sun"].energy = 5

    light = scene.objects["Sun"]
    light_constraint = light.constraints.new("TRACK_TO")
    light_constraint.target = scene.objects["Empty"]

    light.location[0] = 0
    light.location[1] = 0
    light.location[2] = 6

    sunlight = bpy.data.lights["Sun"]
    sunlight.use_shadow = False


def setup_camera(position):
    cam = scene.objects["Camera"]
    cam.location = position
    cam.data.lens = 50
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.target = scene.objects["Empty"]


if __name__ == '__main__':
    args = parser.parse_args()

    # Clear existing objects in the scene
    reset_scene()

    # Load the .obj file
    for obj_path in glob(os.path.join(args.input_dir, "*.obj")):
        id = os.path.basename(obj_path)[:-4].split("_")[-1]
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj = bpy.context.active_object
        mat = obj.data.materials[0]

        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        try:
            texImage.image = bpy.data.images.load(os.path.join(args.input_dir, f"material_{id}.png"))
        except:
            texImage.image = bpy.data.images.load(os.path.join(args.input_dir, f"material_{id}.jpeg"))
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        # if obj.data.materials:
        #     obj.data.materials[0] = mat
        # obj.data.materials.append(mat)

    # create empty axes for tracking
    bpy.ops.object.add(type="EMPTY")

    # Set render settings (optional)
    scene.render.engine = 'BLENDER_EEVEE'  # or 'BLENDER_EEVEE'
    scene.render.resolution_x = 1080  # Width of the output
    scene.render.resolution_y = 1080  # Height of the output
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"

    # bpy.data.worlds['World'].cycles.sample_as_light = True
    # scene.cycles.samples = 32
    # scene.cycles.device = "GPU"
    # scene.cycles.diffuse_bounces = 1
    # scene.cycles.glossy_bounces = 1
    # scene.cycles.transparent_max_bounces = 3
    # scene.cycles.transmission_bounces = 3
    # scene.cycles.filter_width = 0.01
    # scene.render.film_transparent = True

    # Define the path for the rendered image output
    scene.render.filepath = args.output_dir
    setup_camera(args.camera_position)
    add_lighting()
    
    # Render the scene
    bpy.ops.render.render(write_still=True)