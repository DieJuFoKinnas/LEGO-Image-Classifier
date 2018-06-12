import bpy
import os

# TODO: set correct resolution
# TODO:reminder:focus pi camera!!!!
# camera intrinsics for Pi Camera v2
# Sensor size: 3.674 x 2.760 mm (1/4" format in 4:3)
# Lens: f=3.04 mm, f/2.0
# Angle of View: 62.2 x 48.8 degrees
# Full-frame SLR lens equivalent: 29 mm
pi_cam_specs = { "lens": 3.04, "sensor_width": 3.674, "sensor_height": 2.760,
                 "resolution_x": 256, "resolution_y": 144}

# camera intrinsics for Pi Camera v1
# Sensor size: 3.67 x 2.74 mm (1/4" format in 4:3)
# Lens: f=3.6 mm, f/2.9
# Angle of View: 54 x 41 degrees
# Full-frame SLR lens equivalent: 35 mm
pi_cam_v1_specs = { "lens": 3.6, "sensor_width": 3.67, "sensor_height": 2.74,
                    "resolution_x": 256, "resolution_y": 144 }

scene = bpy.context.scene


def get_area(type):
    for area in bpy.context.screen.areas:
        if area.type == type:
            return area


def clear_everything():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # clear all mesh data
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    # clear all camera data
    for camera in bpy.data.cameras:
         bpy.data.cameras.remove(camera)
    # clear all material data
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)


def load_settings():
    scene.render.engine = 'CYCLES'
    scene.cycles.device =  'GPU'
    path = os.path.dirname(os.path.dirname(bpy.__path__[0]))
    #set units to centimeters
    bpy.ops.script.python_file_run(filepath=os.path.join(path, "presets", "units_length", "centimeters.py"))
    #set framerate to 60fps
    bpy.ops.script.python_file_run(filepath=os.path.join(path, "presets", "framerate", "60.py"))
    #set current frame (it autoupdates)
    bpy.context.scene.frame_set(1)


def load_lego_material():
    lego_material = bpy.data.materials.new(name = 'lego_material')
    # modify m_lego diffuse color
    lego_material.diffuse_color = (.8,.8,0)
    # m_lego.node_tree###https://blender.stackexchange.com/questions/23436/control-cycles-material-nodes-and-material-properties-in-python
    return lego_material


def load_light_material():
    light_material = bpy.data.materials.new(name = "light_material")
    light_material.use_nodes = True
    TreeNodes=light_material.node_tree
    links = TreeNodes.links
    for n in TreeNodes.nodes:#delete output node, to start clean
        TreeNodes.nodes.remove(n)
    n_light_out = TreeNodes.nodes.new('ShaderNodeOutputMaterial')# add output node
    n_light_emission = TreeNodes.nodes.new('ShaderNodeEmission')# add emmission node
    links.new(n_light_emission.outputs[0],n_light_out.inputs[0])# link them together (you can see this in the node editor)

    return light_material


def setup_camera(specs, name):
    # create camera data (can be reused, no..?(for when creating multiple))
    camera_data = bpy.data.cameras.new(name=name)
    camera_data.lens = specs["lens"]
    camera_data.sensor_width = specs["sensor_width"]
    camera_data.sensor_height = specs["sensor_height"]
    # create one camera object
    camera = bpy.data.objects.new(name, camera_data)
    # add to scene and update scene
    bpy.context.scene.objects.link(camera)
    bpy.context.scene.update()
    
    bpy.context.scene.render.resolution_x = specs["resolution_x"]
    bpy.context.scene.render.resolution_y = specs["resolution_y"]

    scene.camera = camera


def add_light(name, light_material, angle, x_position):
    bpy.ops.mesh.primitive_plane_add()
    bpy.context.active_object.name = name
    scene.objects.active.scale = (5,5,5)
    scene.objects.active.rotation_euler = (0, angle, 0)
    scene.objects.active.location = (x_position,0,5)
    scene.objects.active.data.materials.append(light_material)


def add_base_plate():
    area = get_area('VIEW_3D')#this is needed for setting the right context, without it the operation would fail https://blender.stackexchange.com/a/53707
    ctx = bpy.context.copy()
    ctx['area'] = area
    ctx['region'] = area.regions[-1] 
    bpy.ops.view3d.snap_cursor_to_center(ctx)

    bpy.ops.mesh.primitive_plane_add()
    bpy.context.active_object.name = 'base_plate'
    scene.objects.active.scale = (9,5,1)
    bpy.ops.rigidbody.objects_add(type='PASSIVE')
    bpy.context.active_object.rigid_body.friction = 0.99 #more friction => faster settletime => shorter simulation
    bpy.context.active_object.rigid_body.collision_shape = 'MESH'


def generate():
    clear_everything()
    lego_material = load_lego_material()
    light_material = load_light_material()
    add_light("light_1", light_material, 45, 7)
    add_light("light_2", light_material, -45, -7)
    load_settings()
    add_base_plate()
    setup_camera(pi_cam_specs, "pi_cam")
    
    return lego_material
