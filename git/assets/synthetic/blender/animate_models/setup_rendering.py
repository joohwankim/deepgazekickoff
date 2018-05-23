# setup rendering

import bpy
from mathutils import Vector, Matrix, Euler
from math import sin, cos, radians, tan
import numpy as np
import os

D, C = bpy.data, bpy.context

def filename_to_id(fname):
    return fname[:-6].replace('female_','f').replace('male_','m')

def makeEyeGreen():
    #Change the texture image for pupil and iris:
    tree = bpy.data.materials['Material_sclera'].node_tree
    nodes, links = tree.nodes, tree.links
        
    node_pupil_green = nodes.new(type='ShaderNodeTexImage')
    path = os.path.join(base_dir, 'eye', 'tex', 'iris_grey_greenPupil.png')
    img = bpy.data.images.load(path)
    node_pupil_green.image = img
    node_pupil_green.location = 0,0
        
    # add a new 
    node_scelra_black = nodes.new(type='ShaderNodeTexImage')
    path = os.path.join(base_dir, 'eye', 'tex', 'sclera_black.png')
    img = bpy.data.images.load(path)
    node_scelra_black.image = img
    node_scelra_black.location = 20,0
        
    # swap links:
    n_mix_iris_sclera = nodes['Mix_iris_sclera']
    old_link = n_mix_iris_sclera.inputs['Color2'].links[0]
    old_node_pupil = n_mix_iris_sclera.inputs['Color2'].links[0].from_node
    links.remove(old_link)
    links.new(node_pupil_green.outputs['Color'], n_mix_iris_sclera.inputs['Color2'])
                
    old_link = n_mix_iris_sclera.inputs['Color1'].links[0]
    old_node_sclera = n_mix_iris_sclera.inputs['Color1'].links[0].from_node
    links.remove(old_link)
    links.new(node_scelra_black.outputs['Color'], n_mix_iris_sclera.inputs['Color1'])
    
    return old_node_pupil, old_node_sclera
    
def makeEyeIR(old_node_pupil, old_node_sclera):
    
    # restore the textures to the original IR ones:
    tree = bpy.data.materials['Material_sclera'].node_tree
    nodes, links = tree.nodes, tree.links
    n_mix_iris_sclera = nodes['Mix_iris_sclera']
    old_link = n_mix_iris_sclera.inputs['Color2'].links[0]
    links.remove(old_link)
    links.new(old_node_pupil.outputs['Color'], n_mix_iris_sclera.inputs['Color2'])
                
    old_link = n_mix_iris_sclera.inputs['Color1'].links[0]
    links.remove(old_link)
    links.new(old_node_sclera.outputs['Color'], n_mix_iris_sclera.inputs['Color1'])    


def removeEnvLighting():
    
     tree = bpy.data.worlds['World'].node_tree
     nds, lks = tree.nodes, tree.links
     n_world_output = nds['World Output']
     old_link = n_world_output.inputs['Surface'].links[0]
     old_linked_node = n_world_output.inputs['Surface'].links[0].from_node
     # remove link to old color
     lks.remove(old_link)  
     
     return old_linked_node

def restoreEnvLighting(old_env_node):
    
    tree = bpy.data.worlds['World'].node_tree
    nds, lks = tree.nodes, tree.links
    n_world_output = nds['World Output']
    lks.new(old_env_node.outputs['Background'], n_world_output.inputs['Surface'])
    
    
def setupCamera(camera_name):
    
    cam = bpy.data.objects[camera_name]
    #eye_obj = bpy.data.objects["Eye1"]
    #eyelashes_lower_obj = bpy.data.objects["Eyelashes_lower1"]
    #eyelashes_upper_obj = bpy.data.objects["Eyelashes_upper1"]
    #head_obj = bpy.data.objects["Head"]
    #scl_obj = bpy.data.objects["Sclera1"]
    #eye_LookAt = bpy.data.objects['Eye1_lookAt']
    #cam_LookAt = bpy.data.objects['Cam_eye_lookAt']
    #scl_mesh = bpy.data.meshes['Sclera']

    C.scene.camera = cam
    C.scene.cycles.film_transparent = True
    C.scene.cycles.use_cache = False

    # setup camera type for eye-region zoom image
    #cam.data.type = 'PERSP'
    #C.scene.camera.data.lens = 33.075 #mm
    #C.scene.camera.data.angle = radians(90)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 0.35    
    
def setupLighting():

    #bpy.context.scene.objects.active = None

    # deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # selection
    if 'led1' in bpy.data.objects:
        bpy.data.objects['led1'].select = True
    if 'led2' in bpy.data.objects:
        bpy.data.objects['led2'].select = True
    if 'led3' in bpy.data.objects:
        bpy.data.objects['led3'].select = True
    if 'led4' in bpy.data.objects:
        bpy.data.objects['led4'].select = True
    if 'led5' in bpy.data.objects:
        bpy.data.objects['led5'].select = True
    # remove selected lights
    bpy.ops.object.delete() 

    # Create new point lamp datablock
    led_data = bpy.data.lamps.new(name="led", type='POINT')
    led_data.shadow_soft_size = 0.02
    led_data.use_nodes = 1
    led_data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 10.0

    # Create new object with our lamp datablock
    lamp1_object = bpy.data.objects.new(name="led1", object_data=led_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lamp1_object)

    lamp2_object = bpy.data.objects.new(name="led2", object_data=led_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lamp2_object)

    lamp3_object = bpy.data.objects.new(name="led3", object_data=led_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lamp3_object)

    lamp4_object = bpy.data.objects.new(name="led4", object_data=led_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lamp4_object)

    lamp5_object = bpy.data.objects.new(name="led5", object_data=led_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lamp5_object)

    '''
    ## Create new area lamp datablock
    lampArea_data = bpy.data.lamps.new(name="arealight", type='AREA')
    # Create new object with our lamp datablock
    lampArea_object = bpy.data.objects.new(name="arealight1", object_data=lampArea_data)
    # Link lamp object to the scene so it'll appear in this scene
    C.scene.objects.link(lampArea_object)
    # And finally select it make active
    lampArea_object.select = True
    C.scene.objects.active = lampArea_object
    lampArea_object.rotation_euler.rotate_axis("X", radians(270))
    lampArea_object.scale = (50.0, 50.0, 50.0)
    lampArea_object.location = (-0.765001, 2.0432, 1.10926)
    '''



    # place the point lamp slightly above the camera:
    t = bpy.data.objects['hmd'].matrix_world.translation
    lamp1_object.location = t

    t = bpy.data.objects['hmd'].matrix_world.translation
    lamp2_object.location = t

    t = bpy.data.objects['hmd'].matrix_world.translation
    lamp3_object.location = t

    t = bpy.data.objects['hmd'].matrix_world.translation
    lamp4_object.location = t

    t = bpy.data.objects['hmd'].matrix_world.translation
    lamp5_object.location = t


    #lamp1_object.location = cam.location + Vector((0.4, 1.25, 0.4))

    # and the area lamp at the locaton camera:
    #lampArea_object.location = eye_obj.location + Vector((0.0, 0.5, 0.0))    
    
def setupRendering(renderengine):    

    if (renderengine == 'GPU'):
        # use the GPU for rendering:
        #C.user_preferences.system.compute_device_type = 'CUDA'
        prefs = bpy.context.user_preferences.addons['cycles'].preferences
        for d in prefs.devices:
            print(d.name)
        prefs.compute_device_type = 'CUDA'

    if (renderengine == 'CPU'):
        C.scene.render.tile_x = C.scene.render.tile_y = 32
        C.scene.render.threads_mode = 'FIXED'
        C.scene.render.threads = 12

    # prepare filename
    this_fname = bpy.path.basename(D.filepath)
    subj_id = filename_to_id(this_fname)
    #base_dir = os.path.realpath('//')
    output_base_dir = 'Z:\\output\\' # This is where output files are stored.
    output_path = os.path.join(output_base_dir, 'output', subj_id)

    # setup image settings

    #C.scene.render.resolution_x = 240
    #C.scene.render.resolution_y = 320
    C.scene.render.resolution_x = 320
    C.scene.render.resolution_y = 569
    C.scene.render.image_settings.file_format='PNG'
    C.scene.render.image_settings.compression = 0
    C.scene.render.resolution_percentage = 100

    bpy.data.scenes["Scene"].render.use_border = True      # Tell Blender to use border render
    bpy.data.scenes["Scene"].render.border_min_y = 0.2890625
    bpy.data.scenes["Scene"].render.border_max_y = 0.7109375
    bpy.data.scenes["Scene"].render.border_min_x = 0.0
    bpy.data.scenes["Scene"].render.border_max_x = 1.0

#img_processed = img_org.crop((555, 0, 1365, 1080)).resize((240,320)).rotate(270,expand='True')

setup_rendering = True
setup_camera = False
set_lighting = False    
renderengine = 'CPU'

camera_name = 'eyetrackingcam_left'

### ----------------------------------------------------------------
### RENDERING SETTINGS
### ----------------------------------------------------------------

if (setup_rendering):
    setupRendering(renderengine)

### ----------------------------------------------------------------
### SETUP CAMERA
### ----------------------------------------------------------------

if (setup_camera):
    setupCamera(camera_name)

### ----------------------------------------------------------------
### SETUP LEDS
### ----------------------------------------------------------------

if (set_lighting):
    setupLighting()

C.scene.update()
