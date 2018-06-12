# This script is intended to be used from inside blender. You can debug it easily by using os.chdir into the directory that contains this script by importing it, and then executing its functions manually.
import bpy
import os
from random import uniform
import math
import numpy as np
from mathutils import Quaternion, Vector

import generate_base_scene


max_piece_count = 1
renders_per_model = 1
camera_angle_count = 1

models_path = "/home/adrian/Downloads/ldraw/parts/"
render_path = "/home/adrian/Projects/JuFoLego/renders/"

scene = bpy.context.scene


def model_name(filename):
    return filename.split('.')[0]


def remove_object(object):
    bpy.ops.object.select_all(action='DESELECT')
    object.select = True
    bpy.ops.object.delete()


def load_piece(model_path):
    bpy.ops.import_scene.importldraw(filepath=model_path)
    # bpy.ops.import_mesh.stl(filepath = lego_piece_code +".stl")#no ovearhead test
    
    for object in bpy.context.scene.objects:
        if ".dat" in object.name:
            lego_piece = object

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')

    ground_plane = bpy.context.scene.objects["LegoGroundPlane"]
    remove_object(ground_plane)

    # TODO: I don't know whether I need to join the piece
    # TODO: not sure if the default material is fine
    return lego_piece


# axis_x, ... are the components of the rotation axis. The axis vector should have a norm of 1.
def quaternion_from_rotation(angle, axis_x, axis_y, axis_z):
    sine = math.sin(angle/2)
    return Quaternion([math.cos(angle/2), sine * axis_x, sine * axis_y, sine * axis_z])


# note that current angle is expected to be an integer
def position_camera(camera, camera_angle_count, current_angle, height=5, radius=3):
    # place cammera somewhere on an imaginary circle and then rotate it to the origin
    z_angle = 2 * math.pi * current_angle / camera_angle_count
    camera.location = (radius * math.cos(z_angle), radius * math.sin(z_angle), height)
    
    # using quaternions here, since you can easily concatenate their rotations
    base_rotation = quaternion_from_rotation(math.pi/2, 0, 0, 1)
    
    to_origin = -camera.location
    to_origin.normalize()
    y_angle = math.acos(to_origin * Vector([0, 0, -1])) # angle between vector to origin and vertical vector
    y_rotation = quaternion_from_rotation(y_angle, 0, 1, 0)
    z_rotation = quaternion_from_rotation(z_angle, 0, 0, 1)
    
    camera.rotation_quaternion = z_rotation * y_rotation * base_rotation


def position_piece(lego_piece):
    diagonal_length = np.linalg.norm(np.array(lego_piece.dimensions))
    lego_piece.location = (0, 0, diagonal_length / 2 + 0.1)
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


def simulate_drop(lego_piece, epsilon=5.0e-06, min_frames=20, max_frames=5000):
    # set current frame and autoupdates
    scene.frame_set(1)
    
    old_matrix = np.array(lego_piece.matrix_world)
    current_matrix = np.array(lego_piece.matrix_world)
    diff = 1.0
    
    # keep the animation going until the piece doesn't move anymore
    while((diff > epsilon or scene.frame_current < min_frames) and scene.frame_current < max_frames):
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


def debug_load(i, lego_material):
    path = os.path.join(models_path,os.listdir(models_path)[i])
    lego_piece = load_piece(path)
    position_piece(lego_piece)
    make_rigid(lego_piece, lego_material)
    
    return lego_piece


if __name__ == "__main__":
    lego_material = generate_base_scene.generate()

    filenames = os.listdir(models_path)[:max_piece_count]
    all_paths = [os.path.join(models_path, filename) for filename in filenames]
    model_paths = [path for path in all_paths if not os.path.isdir(path)]

    for model_path in model_paths:
        lego_piece = load_piece(model_path)
        
        # for some reason importldraw overwrites this every time
        camera.rotation_mode = "QUATERNION"

        position_piece(lego_piece)
        make_rigid(lego_piece, lego_material)
        for render_number in range(renders_per_model):
            position_piece(lego_piece)
            simulate_drop(lego_piece)
            for current_angle in range(camera_angle_count):
                position_camera(scene.camera, camera_angle_count, current_angle)
                render(render_path, "{}-{}".format(model_name(piece_name), render_number))
            
        remove_object(lego_piece)

