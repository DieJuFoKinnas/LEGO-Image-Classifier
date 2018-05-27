import bpy
import os
from random import uniform

PI = 3.14159265359

# TODO: auto import a entire directory 

########### CONFIG ###########
render = True
#render = False
lego_piece_code = '3001'#'32498'#
TIMES_PER_PIECE = 10
scene = bpy.data.scenes['Scene']
scene.frame_end = 80
#TODO:reminder:focus pi camera!!!!
#camara intrinsics for Pi Camera v2
#Sensor size: 3.674 x 2.760 mm (1/4" format in 4:3)
#Lens: f=3.04 mm, f/2.0
#Angle of View: 62.2 x 48.8 degrees
#Full-frame SLR lens equivalent: 29 mm
class PiCam:#used to keep adjustments
    lens = 3.04#mm
    width = 3.674#mm, sensor
    height = 2.760#mm, sensor

#camara intrinsics for Pi Camera v1
#Sensor size: 3.67 x 2.74 mm (1/4" format in 4:3)
#Lens: f=3.6 mm, f/2.9
#Angle of View: 54 x 41 degrees
#Full-frame SLR lens equivalent: 35 mm
class PiCamv1:#used to keep adjustments
    lens = 3.6#mm
    width = 3.67#mm, sensor
    height = 2.74#mm, sensor
########### end of CONFIG ###########

        ###INFO###
    # _o_ -> object
    # _d_ -> data
    # _m_ -> material
    # _n_ -> node 
        ###INFO###

########### SETUP eviroment ###########
#set render
scene.render.engine = 'CYCLES'
scene.cycles.device =  'GPU'
#set units to centimeters
bpy.ops.script.python_file_run(filepath="C:\\Program Files\\Blender Foundation\\Blender\\2.79\\scripts\\presets\\units_length\\centimeters.py")
#set framerate to 60fps
bpy.ops.script.python_file_run(filepath="C:\\Program Files\\Blender Foundation\\Blender\\2.79\\scripts\\presets\\framerate\\60.py")
#set current frame (it autoupdates)
bpy.context.scene.frame_set(1)

# clear all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
#clear all mesh data
for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)
#clear all camera data
for camera in bpy.data.cameras:
     bpy.data.cameras.remove(camera)
# clear all material data
for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)


#create camera data (can be reused, no..?(for when creating multiple))
PiCam_d = bpy.data.cameras.new(name = 'PiCam_d')
PiCam_d.lens = PiCam.lens
PiCam_d.sensor_width = PiCam.width
PiCam_d.sensor_height = PiCam.height
#create one camera object
PiCam_o = bpy.data.objects.new('PiCam_o',PiCam_d)
#add to scene and update scene
bpy.context.scene.objects.link(PiCam_o)
bpy.context.scene.update()
#move camera into position
PiCam_o.location = (0,0,15)#15cm high
scene.camera = PiCam_o#select the camera for rendering

#add ground_plane
#set cursor to the center
for area in bpy.context.screen.areas:#this is needed for setting the right context, without it the operation would fail https://blender.stackexchange.com/a/53707
    if area.type == 'VIEW_3D':
        ctx = bpy.context.copy()
        ctx['area'] = area
        ctx['region'] = area.regions[-1] 
        bpy.ops.view3d.snap_cursor_to_center(ctx)
bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = 'ground_plane_o'
scene.objects.active.scale = (9,5,1)#
bpy.ops.rigidbody.objects_add(type='PASSIVE')
bpy.context.active_object.rigid_body.friction = 0.99 #more friction => faster settletime => shorter simulation
bpy.context.active_object.rigid_body.collision_shape = 'MESH'

#add new material for lights
m_light = bpy.data.materials.new(name = 'light_m')
m_light.use_nodes = True
TreeNodes=m_light.node_tree
links = TreeNodes.links
for n in TreeNodes.nodes:#delete output node, to start clean
    TreeNodes.nodes.remove(n)
n_light_out = TreeNodes.nodes.new('ShaderNodeOutputMaterial')#add output node
n_light_emission = TreeNodes.nodes.new('ShaderNodeEmission')#add emmission node
links.new(n_light_emission.outputs[0],n_light_out.inputs[0])#link them together (you can see this in the node editor)

#add first light
bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = 'light_1_o'
scene.objects.active.scale = (5,5,5)
scene.objects.active.rotation_euler = (0,45,0)
scene.objects.active.location = (7,0,5)
scene.objects.active.data.materials.append(m_light)
#add second light
bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = 'light_2_o'
scene.objects.active.scale = (5,5,5)
scene.objects.active.rotation_euler = (0,-45,0)
scene.objects.active.location = (-7,0,5)
scene.objects.active.data.materials.append(m_light)

#add new material
m_lego = bpy.data.materials.new(name = 'lego_m')
#modifie m_lego diffuse color
m_lego.diffuse_color = (.8,.8,0)
##m_lego.node_tree###https://blender.stackexchange.com/questions/23436/control-cycles-material-nodes-and-material-properties-in-python


#we need to record enviroment objects because the import could come in a bunch of pieces that need to be joined and we use this list to not join the enviroment with the lego piece
enviroment_obj = [obj for obj in bpy.data.objects]

########### end SETUP eviroment ###########


########### BEGIN LOOP ###########
# import lego piece
bpy.ops.import_scene.ldraw(filepath = lego_piece_code +".dat")
#bpy.ops.import_mesh.stl(filepath = lego_piece_code +".stl")#no ovearhead test
bpy.ops.object.select_all(action='SELECT')
for obj in enviroment_obj:
    obj.select = False
bpy.ops.object.join()#sometimes the pice comes in parts, so we join them
lego_piece = scene.objects.active
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
#iterate through TIMES_PER_PIECE positions
for piece_number in range(TIMES_PER_PIECE):
    lego_piece.location = (0,0,max(lego_piece.dimensions)/2)
    #assign material m_lego to lego_piece
    lego_piece.data.materials.append(m_lego)

    #random rotation
    lego_piece.rotation_euler = (uniform(0,PI),uniform(0,PI),uniform(0,PI))

    #add lego_piece to rigid body
    bpy.ops.object.select_all(action='DESELECT')
    lego_piece.select = True
    bpy.ops.rigidbody.objects_add(type='ACTIVE')
    bpy.context.active_object.rigid_body.friction = 0.99
    bpy.context.active_object.rigid_body.collision_shape = 'CONVEX_HULL'

    #set current frame and autoupdates
    bpy.context.scene.frame_set(1)
    
    #calculate position of piece by advancing frame by frame and locking at the change in position/rotation until the framedifference is minimal (piece is then settled)
    new_matrix = lego_piece.matrix_world.copy()#contains translation, rotation and scale
    old_matrix = lego_piece.matrix_world.copy()
    while(True):
        scene.frame_set(bpy.context.scene.frame_current + 1)#advance frame by one
        new_matrix = lego_piece.matrix_world.copy()
        #to know if the piece is settled down we compute change of position and rotation by tacking the difference, we also keep track of the greatest change
        maxi = 0
        for i in range(4):
            for j in range(4):
                temp = abs(old_matrix[i][j]) - abs(new_matrix[i][j])
                if(maxi<temp):
                    maxi = temp
        if (maxi<5.0e-07 and 20<scene.frame_current):# if maxi(greatest change) is smaler than an trivial threshold and the current frame is at least the 21 (this is because at impact/first aceleration the speed is small and we let the piece acelerate)
            break
        else:
            old_matrix = new_matrix.copy()
            scene.rigidbody_world.point_cache.frame_end = scene.frame_current + 1 # increase the rigid body buffer size for next step
            
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)#HACK: update window
    
    if render:
        print("Rendering now")
        scene.render.filepath = '//renders/' + str(lego_piece_code) + "_" + str(piece_number) + '.png'
        bpy.ops.render.render(write_still=True)
        print("finished rendering " + scene.render.filepath)
########### END LOOP ###########

print('FINISHED')