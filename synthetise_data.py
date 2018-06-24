# This script is intended to be used from inside blender. You can debug it easily by using os.chdir into the directory that contains this script by importing it, and then executing its functions manually.
import bpy
import time
import os
from random import uniform, gauss
import math
import sys
import numpy as np
from mathutils import Quaternion, Vector

import generate_base_scene


simulations_per_model = 1
camera_angle_count = 1
piece_count = 100
debug_mode = True

models_path = "/home/adrian/Downloads/ldraw/parts/"
render_path = "renders/"

scene = bpy.context.scene


def drop_extension(filename):
    return str.split(filename, ".")[0]


def get_paths():
    filenames = os.listdir(models_path)
    all_paths = [(filename, os.path.join(models_path, filename)) for filename in filenames]
    all_models = [(drop_extension(name), path) for name, path in all_paths if not os.path.isdir(path)]
    # Yes, I'm racist and I'm excluding half of the parts simply because they have letters in their name.
    # In the future the list of rendered models can be extended to be more exhaustive, though thats
    # not required for now since ~5000 pieces is enough
    acceptable_models = [(name, path) for name, path in all_models if str.isnumeric(name)]
    
    return acceptable_models


def remove_object(object):
    bpy.ops.object.select_all(action='DESELECT')
    object.select = True
    bpy.ops.object.delete()


def load_piece(model_path):
    print("loading {}".format(model_path))
    bpy.ops.import_scene.importldraw(filepath=model_path)
    # bpy.ops.import_mesh.stl(filepath = lego_piece_code +".stl")#no ovearhead test

    components = [c for c in bpy.context.scene.objects if ".dat" in c.name]

    ground_plane = bpy.context.scene.objects["LegoGroundPlane"]
    remove_object(ground_plane)

    # for some reason importldraw overwrites this every time
    scene.camera.rotation_mode = "QUATERNION"
    
    return components

# I do not know how this affects material and reder properties, since in reality a piece could consist
# of multiple materials
def join_components(components):
    if len(components) == 1:
        bpy.ops.object.select_all(action='DESELECT')
        components[0].select = True
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
        return components[0]
    else:
        for c in components:
            c.select = True
        
        piece = [c for c in components if c.type == "MESH"][0]
        scene.objects.active = piece
        bpy.ops.object.join()
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
        
        joined = piece.copy()

        parents = [c for c in components if c.type != "MESH" and not c.parent]
        joined.name = parents[0].name
        
        [remove_object(c) for c in components]
        scene.objects.link(joined)
        return joined
    


# axis_x, ... are the components of the rotation axis. The axis vector should have a norm of 1.
def quaternion_from_rotation(angle, axis_x, axis_y, axis_z):
    sine = math.sin(angle/2)
    return Quaternion([math.cos(angle/2), sine * axis_x, sine * axis_y, sine * axis_z])


# note that current angle is expected to be an integer
def position_camera(camera, height=4, radius=3, std_dev=0.4):
    height = height + gauss(0, std_dev)
    radius = height + gauss(0, std_dev)
    
    # place cammera somewhere on an imaginary circle and then rotate it to the origin
    z_angle = uniform(0, 2*math.pi)
    camera.location = (radius * math.cos(z_angle), radius * math.sin(z_angle), height)
    
    # using quaternions here, since you can easily concatenate their rotations
    base_rotation = quaternion_from_rotation(math.pi/2, 0, 0, 1)
    
    to_origin = -camera.location
    to_origin.normalize()
    y_angle = math.acos(to_origin * Vector([0, 0, -1])) # angle between vector to origin and vertical vector
    y_rotation = quaternion_from_rotation(y_angle, 0, 1, 0)
    z_rotation = quaternion_from_rotation(z_angle, 0, 0, 1)
    
    camera.rotation_quaternion = z_rotation * y_rotation * base_rotation


def position_piece(lego_piece, min_height=0.2, average_height=0.3, std_dev=0.4):
    diagonal_length = np.linalg.norm(np.array(lego_piece.dimensions)) # placing it makes it sometimes appear inside the ground
    lego_piece.location = (gauss(0, std_dev), gauss(0, std_dev), diagonal_length)
    lego_piece.rotation_euler = uniform(0, 2*math.pi),uniform(0, 2*math.pi),uniform(0, 2*math.pi)


def make_rigid(lego_piece, lego_material):
    lego_piece.data.materials.append(lego_material)

    # you have to also make the piece active so it lets you add the rigidbody
    bpy.context.scene.objects.active = lego_piece
    bpy.ops.object.select_all(action='DESELECT')
    lego_piece.select = True

    bpy.ops.rigidbody.objects_add(type='ACTIVE')
    bpy.context.active_object.rigid_body.friction = 0.99
    bpy.context.active_object.rigid_body.collision_shape = 'CONVEX_HULL'


def distance_from_origin(object):
    x, y, _ = object.matrix_world * Vector([0, 0, 0])
    return np.linalg.norm(np.array([x, y]))


def simulate_drop(lego_piece, epsilon=5.0e-06, min_frames=20, max_frames=5000, max_distance_from_origin=1.8):
    # set current frame and autoupdates
    scene.frame_set(1)
    
    old_matrix = np.array(lego_piece.matrix_world)
    current_matrix = np.array(lego_piece.matrix_world)
    diff = 1.0
    
    # keep the animation going until the piece doesn't move anymore or it is about to roll out of the frame
    while(diff > epsilon and scene.frame_current < max_frames and distance_from_origin(lego_piece) < max_distance_from_origin):
        scene.frame_set(bpy.context.scene.frame_current + 1)
        
        old_matrix = current_matrix.copy()
        current_matrix = np.array(lego_piece.matrix_world)
        diff = np.linalg.norm(old_matrix - current_matrix)

        scene.rigidbody_world.point_cache.frame_end = scene.frame_current + 1 # increase the rigid body buffer size for next step
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)#HACK: update window


def render(render_path, shot_name):
    context = bpy.context

    print("Rendering ", model_path)
    context.scene.render.filepath = os.path.join(render_path, shot_name + '.jpeg')
    bpy.ops.render.render(write_still=True)


# for testing purposes: 4767.dat is has multiple parts and 480.dat rolls
# TODO: when we scale this to more pieces, huge pieces like 6024.dat should be excluded
# TODO: when we scale this, sticker-pieces should be exclude(either excluding the piece numbers corresponding to stickers
# (this is documented in the ldraw piece numbering FAQ) or by checking for a completely flat z-dimension(probably easier))
# use this for testing purposes by importing it in the console
def debug_load(name):
    lego_material = generate_base_scene.generate()
    lego_piece = join_components(load_piece("{}/{}".format(models_path, name)))
    position_camera(scene.camera)
    position_piece(lego_piece)
    make_rigid(lego_piece, lego_material)
    simulate_drop(lego_piece)
    
    ctx = generate_base_scene.get_view3d_context()
    bpy.ops.view3d.viewnumpad(ctx, type='CAMERA')

    return lego_piece


if __name__ == "__main__":
    lego_material = generate_base_scene.generate()
    
    position_camera(scene.camera, 10, 1)

    for i, (piece_name, model_path) in list(enumerate(get_paths()))[:piece_count]:
        components = load_piece(model_path)
        lego_piece = join_components(components)

        position_piece(lego_piece)
        make_rigid(lego_piece, lego_material)
        for simulation_number in range(simulations_per_model):
            position_piece(lego_piece)
            simulate_drop(lego_piece)
            for current_angle in range(camera_angle_count):
                position_camera(scene.camera)
                render(render_path, "{}-{}".format(drop_extension(piece_name), simulation_number))

            if debug_mode:
                input("press enter for loading the next piece(currently at piece {})".format(i))
            
        remove_object(lego_piece)

