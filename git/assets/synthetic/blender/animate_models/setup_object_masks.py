import bpy, pdb, os
import numpy as np

import imp
import NXPModelPrepHelper
imp.reload(NXPModelPrepHelper)
from NXPModelPrepHelper import *

def createObjectMaskScene():

#############################################################################
# object id scene for all the objects
#############################################################################
    deselect_all_objects()
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.scene.new(type='LINK_OBJECTS')
    # rename the new scene with 'objectids'
    bpy.data.scenes['Scene.001'].name = "objectids"
    bpy.data.screens['Default'].scene = bpy.data.scenes['objectids']

    # duplicate face
    obj = bpy.data.objects['head_without_eye']
    obj2 = obj.copy() # duplicate linked
    obj2.data = obj.data.copy() # optional: make this a real duplicate (not linked)
    obj2.name = 'head_without_eye_mask'
    bpy.context.scene.objects.unlink(obj)
    bpy.context.scene.objects.link(obj2) # add to scene
    
    # sclera not needed for this scene.
    obj = bpy.data.objects['left_sclera']
    obj2 = obj.copy() # duplicate linked
    obj2.data = obj.data.copy() # optional: make this a real duplicate (not linked)
    obj2.name = 'left_sclera_black'
    bpy.context.scene.objects.unlink(obj)
    bpy.context.scene.objects.link(obj2)
    
    # duplicate eyelashes lower
    obj = bpy.data.objects['left_lower_eyelashes']
#    obj2 = obj.copy() # duplicate linked
#    obj2.data = obj.data.copy() # optional: make this a real duplicate (not linked)
#    obj2.name = 'left_lower_eyelashes_mask'
    bpy.context.scene.objects.unlink(obj)
#    bpy.context.scene.objects.link(obj2) # add to scene
    
    # duplicate eye lashes upper
    obj = bpy.data.objects['left_upper_eyelashes']
#    obj2 = obj.copy() # duplicate linked
#    obj2.data = obj.data.copy() # optional: make this a real duplicate (not linked)
#    obj2.name = 'left_upper_eyelashes_mask'
    bpy.context.scene.objects.unlink(obj)
#    bpy.context.scene.objects.link(obj2) # add to scene

    # duplicate leds and color-code them. remove the original object from the objectid scene.
    for ii in np.arange(1,27,1):
        led_str = 'led_' + str(ii).zfill(3)
        obj = bpy.data.objects[led_str]
        obj2 = obj.copy()
        obj2.data = obj.data.copy()
        obj2.name = led_str + '_mask'
        bpy.context.scene.objects.link(obj2)
        bpy.context.scene.objects.unlink(bpy.data.objects[led_str])
    
#############################################################################
# object id scene without the head (for unoccluded pupil and iris)
#############################################################################
    deselect_all_objects()
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.scene.new(type='LINK_OBJECTS')
    # rename the new scene with 'objectids'
    bpy.data.scenes['objectids.001'].name = "objectids_noskin"
    bpy.data.screens['Default'].scene = bpy.data.scenes['objectids_noskin']

    # unlink all head-related objects from the scene.
    #bpy.context.scene.objects.unlink(bpy.data.objects['head_without_eye'])
    #bpy.context.scene.objects.unlink(bpy.data.objects['left_sclera'])
    #bpy.context.scene.objects.unlink(bpy.data.objects['left_lower_eyelashes'])
    #bpy.context.scene.objects.unlink(bpy.data.objects['left_upper_eyelashes'])
    bpy.context.scene.objects.unlink(bpy.data.objects['head_without_eye_mask'])
    bpy.context.scene.objects.unlink(bpy.data.objects['left_sclera_black'])
    #bpy.context.scene.objects.unlink(bpy.data.objects['left_lower_eyelashes_mask'])
    #bpy.context.scene.objects.unlink(bpy.data.objects['left_upper_eyelashes_mask'])

    # duplicate sclera
    obj = bpy.data.objects['left_sclera']
    obj2 = obj.copy() # duplicate linked
    obj2.data = obj.data.copy() # optional: make this a real duplicate (not linked)
    obj2.name = 'left_sclera_mask'
    bpy.context.scene.objects.link(obj2) # add to scene

    # unlink all LEDs.
    for ii in np.arange(1,27,1):
        led_str = 'led_' + str(ii).zfill(3) + '_mask'
        bpy.context.scene.objects.unlink(bpy.data.objects[led_str])

    # We only need sclera for this one.
    #bpy.context.scene.objects.link(bpy.data.objects['left_sclera_mask']) # add to scene # not needed to add - it should be already there.
    
def createObjectMaskMaterials():

    # create new material for face
    skinIdMat = bpy.data.materials.new(name="SkinObjectID")
    skinIdMat.use_nodes = True
    #bpy.ops.node.add_node(type="ShaderNodeEmission", use_transform=True)
    # Remove default
    skinIdMat.node_tree.nodes.remove(skinIdMat.node_tree.nodes.get('Diffuse BSDF'))
    material_output = skinIdMat.node_tree.nodes.get('Material Output')
    emission = skinIdMat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 0.2
    emission.inputs[0].default_value = (1, 0, 0, 1)
    # link emission shader to material
    skinIdMat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])
    bpy.data.objects['head_without_eye_mask'].data.materials[0] = skinIdMat

    # create new material for face
    blackScleraIdMat = bpy.data.materials.new(name="BlackScleraMaterial")
    blackScleraIdMat.use_nodes = True
    #bpy.ops.node.add_node(type="ShaderNodeEmission", use_transform=True)
    # Remove default
    blackScleraIdMat.node_tree.nodes.remove(blackScleraIdMat.node_tree.nodes.get('Diffuse BSDF'))
    material_output = blackScleraIdMat.node_tree.nodes.get('Material Output')
    emission = blackScleraIdMat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 0.0
    emission.inputs[0].default_value = (0, 0, 0, 0)
    # link emission shader to material
    blackScleraIdMat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])
    bpy.data.objects['left_sclera_black'].data.materials[0] = blackScleraIdMat

    for ii in np.arange(1,27,1):
        # create new material for led
        if ii == 1:
            bpy.data.objects['led_'+str(ii).zfill(3)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (0,1,0,1)
        elif ii == 6:
            bpy.data.objects['led_'+str(ii).zfill(3)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (0,0,1,1)
        elif ii == 16:
            bpy.data.objects['led_'+str(ii).zfill(3)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (1,1,0,1)
        elif ii == 19:
            bpy.data.objects['led_'+str(ii).zfill(3)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (1,0,1,1)
        else: # leave all other leds cyan-colored
            bpy.data.objects['led_'+str(ii).zfill(3)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (0,1,1,1)
        #elif ii == 10:
        #    bpy.data.objects['led'+str(ii)+'_mask'].data.node_tree.nodes['Emission'].inputs[0].default_value = (0,1,1,1)
        # # link emission shader to material
        # led_mat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])
        # bpy.data.objects[led_obj_name].data.materials[0] = led_mat

    # create new material for sclera
    scleraIdMat = bpy.data.materials.new(name="ScleraObjectID")
    scleraIdMat.use_nodes = True
    bpy.data.objects['left_sclera_mask'].data.materials[0] = scleraIdMat
    # Remove default
    scleraIdMat.node_tree.nodes.remove(scleraIdMat.node_tree.nodes.get('Diffuse BSDF'))
    # setup  texture    
    texcoord = scleraIdMat.node_tree.nodes.new('ShaderNodeTexCoord')
    texcoord.location = 0,0
    imagetex = scleraIdMat.node_tree.nodes.new('ShaderNodeTexImage')
    imagetex.location = 200,0
    imagedirectory = os.path.join(os.path.dirname(os.path.dirname(bpy.data.filepath)),'tex')
    imagepath = os.path.join(imagedirectory,'eye_object_ids_combined3.png')
    imagename = 'eye_object_ids_combined3.png'
    #bpy.ops.image.open(filepath=imagepath, directory=imagedirectory, files=[{"name":imagename, "name":imagename}], show_multiview=False)
    bpy.data.images.load(imagepath)
    imagetex.image = bpy.data.images[imagename]
    emission = scleraIdMat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 0.5
    emission.inputs[0].default_value = (1, 1, 0, 1)
    emission.location = 400,0
    material_output = scleraIdMat.node_tree.nodes.get('Material Output')
    material_output.location = 600,0
    # link emission shader to material
    scleraIdMat.node_tree.links.new(imagetex.inputs[0], texcoord.outputs[2])
    scleraIdMat.node_tree.links.new(emission.inputs[0], imagetex.outputs[0])
    scleraIdMat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])
    
    # create new material for eye lashes
    eyeLashesIdMat = bpy.data.materials.new(name="EyeLashesObjectID")
    eyeLashesIdMat.use_nodes = True
#    bpy.data.objects['left_lower_eyelashes_mask'].data.materials[1] = eyeLashesIdMat
#    bpy.data.objects['left_upper_eyelashes_mask'].data.materials[1] = eyeLashesIdMat
    eyeLashesIdMat.node_tree.nodes.remove(eyeLashesIdMat.node_tree.nodes.get('Diffuse BSDF'))
    material_output = eyeLashesIdMat.node_tree.nodes.get('Material Output')
    emission = eyeLashesIdMat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 0.5
    emission.inputs[0].default_value = (0.5, 1, 1, 1)
    emission.location = 400,0
    # link emission shader to material
    eyeLashesIdMat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])

def setupCompositor(output_path):
    # activate default scene
    bpy.data.screens['Default'].scene = bpy.data.scenes['Scene']

    bpy.data.scenes["Scene"].use_nodes = True
    bpy.data.scenes['Scene'].render.filepath = os.path.join(output_path.replace('.blend',''),'type_img_frame_')
    bpy.data.scenes['objectids'].render.filepath = os.path.join(output_path.replace('.blend',''),'type_maskWithSkin_frame_')
    bpy.data.scenes["objectids"].render.image_settings.color_mode = 'RGB'

    tree = bpy.context.scene.node_tree

    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # create input image node
 #   image_node = tree.nodes.new(type='CompositorNodeImage')
#    image_node.image = bpy.data.images['YOUR_IMAGE_NAME']
    #image_node.location = 0,0
    
    # create render layer node
    rl_node_objectid = tree.nodes.new(type='CompositorNodeRLayers')
    rl_node_objectid.location = 0,0
    rl_node_objectid.scene = bpy.data.scenes['objectids']    
    
    rl_node = tree.nodes.new(type='CompositorNodeRLayers')
    rl_node.location = 0,350
    rl_node.scene = bpy.data.scenes['Scene']
    
    rl_node_objectid_noskin = tree.nodes.new(type='CompositorNodeRLayers')
    rl_node_objectid_noskin.location = 0,700
    rl_node_objectid_noskin.scene = bpy.data.scenes['objectids_noskin']
    
    file_output_node = tree.nodes.new(type = 'CompositorNodeOutputFile')
    file_output_node.location = 300, 300
    file_output_node.file_slots.clear()
    file_output_node.file_slots.new('type_img_frame_')
    file_output_node.file_slots.new('type_maskWithSkin_frame_')
    file_output_node.file_slots.new('type_maskWithoutSkin_frame_')
    file_output_node.base_path = output_path.replace('.blend','')

    link = tree.links.new(rl_node.outputs[0],file_output_node.inputs['type_img_frame_'])
    link = tree.links.new(rl_node_objectid.outputs[0],file_output_node.inputs['type_maskWithSkin_frame_'])
    link = tree.links.new(rl_node_objectid_noskin.outputs[0],file_output_node.inputs['type_maskWithoutSkin_frame_'])
    file_output_node.format.color_mode = 'RGB'
    file_output_node.file_slots['type_img_frame_'].use_node_format = False
    file_output_node.file_slots['type_img_frame_'].format.color_mode = 'BW'
    file_output_node.file_slots['type_maskWithSkin_frame_'].use_node_format = False
    file_output_node.file_slots['type_maskWithSkin_frame_'].format.color_mode = 'RGB'
    file_output_node.file_slots['type_maskWithoutSkin_frame_'].use_node_format = False
    file_output_node.file_slots['type_maskWithoutSkin_frame_'].format.color_mode = 'RGB'

#    # create multiply node for grayscale scene -> multiply with blue channel
#    mix_objectid_node = tree.nodes.new(type='CompositorNodeMixRGB')
#    mix_objectid_node.location = 300,0
#    mix_objectid_node.blend_type = 'MULTIPLY'
#    mix_objectid_node.inputs[2].default_value = (1, 1, 0, 1)
#    link = tree.links.new(rl_node_objectid.outputs[0], mix_objectid_node.inputs[1])
#    
#    # create multiply node for object mask scene -> multiply with yellow
#    mix_scene_node = tree.nodes.new(type='CompositorNodeMixRGB')
#    mix_scene_node.location = 300,350
#    mix_scene_node.blend_type = 'MULTIPLY'
#    mix_scene_node.inputs[2].default_value = (0, 0, 1, 1)
#    link = tree.links.new(rl_node.outputs[0], mix_scene_node.inputs[1])
#    
#    # create add node to combine both outputs
#    add_node = tree.nodes.new(type='CompositorNodeMixRGB')
#    add_node.location = 600,200
#    add_node.blend_type = 'ADD'
#    link = tree.links.new(mix_scene_node.outputs[0], add_node.inputs[1])
#    link = tree.links.new(mix_objectid_node.outputs[0], add_node.inputs[2])
#    
#    # combine output node of last node with composite node input
#    comp_node = tree.nodes.new('CompositorNodeComposite')   
#    comp_node.location = 800,200
#    link = tree.links.new(add_node.outputs[0], comp_node.inputs[0])

def setup_object_masks(blendfile_path, output_path):
    bpy.ops.wm.open_mainfile(filepath=blendfile_path)
    createObjectMaskScene()
    createObjectMaskMaterials()
    setupCompositor(output_path)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath = blendfile_path)
