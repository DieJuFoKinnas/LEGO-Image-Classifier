
import bpy
import os


models_path = "/home/adrian/Data/ldraw/parts/"
render_path = "/home/adrian/Projects/JuFoLego/renders/"


def model_name(filename):
    return filename.split('.')[0]


def init():
    context = bpy.context
    context.screen.scene = bpy.data.scenes['Base']

    context.scene.camera = context.scene.objects['Camera']
    # make a new scene with cam, lights and world linked
    bpy.ops.scene.new(type='LINK_OBJECTS')
    context.scene.name = "Model"
    bpy.context.scene.cycles.device = 'GPU'


def load_piece(model_path):
    # importing a piece changes camera settings, so they are stored so they can
    # be applied later
    context = bpy.context
    cam = context.scene.camera
    # the slice operator creates a copy, not a reference to a mutable object
    cam_location = cam.location[:]
    cam_rotation = cam.rotation_euler
    bpy.ops.import_scene.importldraw(filepath=model_path)
    cam.location = cam_location
    cam.rotation_euler = cam_rotation


def render(render_path, model_name):
    context = bpy.context

    print("Render ", model_path)
    context.scene.render.filepath = os.path.join(render_path, model_name + '.jpeg')
    bpy.ops.render.render(write_still=True)


def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    # ldraw names objects like filenames, so we can easily find leftover pieces
    # from previous renders
    for piece in bpy.context.scene.objects:
        if ".dat" in piece.name:
            piece.select = True
            bpy.ops.object.delete()


init()
for piece in os.listdir(models_path)[17:18]:
    model_path = os.path.join(models_path, piece)
    clear_scene()
    load_piece(model_path)
    render(render_path, model_name(piece))


#load_piece(os.path.join(models_path, "31c.dat"))
