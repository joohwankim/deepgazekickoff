import bpy
import numpy as np
import os
from math import degrees, radians
from random import gauss
        
def pose_eyelids_animated(pitch, eyeOpenness, random=False, animate=True):

#    #if random: pitch = pitch + gauss(0,5)
#    if random: pitch = pitch + gauss(0,2)

#    # calculate how much to look up and down
#    lu_max, ld_max = 20, -25.0
#    ld_amt = (1-(ld_max-pitch)/ld_max) if (pitch < 0) else 0
#    lu_amt = (1-(lu_max-pitch)/lu_max) if (pitch > 0) else 0

#    eyeClosedness = 1.0 - eyeOpenness
#    ld_amt = max(ld_amt,eyeClosedness) # eye can only be further closed

    # Custom formula for female_01.
    # Gaze direction, minimum, maximum (- means look down + means look up)
    # 24 deg, -0.145, +1.0
    # 20 deg, -0.246, +1.0
    # 10 deg, -0.400, +1.0
    #  0 deg, -0.575, +0.5
    #-10 deg, -0.750, +0.0
    #-20 deg, -0.915, -0.1
    #-24 deg, -0.986, -0.2
    # Three observations
    # 1. Minimum values are almost linear as a function of gaze direction (y = 0.0172 * x - 0.5739)
    # 2. Maximum values are piece-wise linear (one line for negative values and another for positive values) with saturation at 1
    # 3. Variation in positive values result in much more subtle change in the rendered scene. About 10% in my impression.
    #    So compress positive region to 1/10 during sampling and expand it when assigning to lu_amt and ld_amt.
    l_max = (pitch + 10) * 0.005 if (pitch > -10) else (pitch + 10) * 0.01 # 0.05 became 0.005 for the compression in positive region
    l_max = 0.1 if (l_max > 0.1) else l_max # again compression to 1/10
    l_min = 0.0172 * pitch - 0.5739
    l_value = l_min + (l_max - l_min) * eyeOpenness
    lu_amt = l_value if (l_value > 0) else 0
    ld_amt = -l_value if (l_value < 0) else 0

    # get meshes
    lids = bpy.data.meshes['left_eyelid_mesh']
    upper_lashes = bpy.data.meshes['left_upper_eyelashes_mesh']
    lower_lashes = bpy.data.meshes['left_lower_eyelashes_mesh']
    head = bpy.data.meshes['head_mesh']

    # set shape keys
    for mesh in [lids, upper_lashes, lower_lashes, head]:
        mesh.shape_keys.key_blocks['Look_down'].value = ld_amt
        mesh.shape_keys.key_blocks['Look_up'].value = lu_amt
    

    #mesh.shape_keys.key_blocks['Look_up'].keyframe_insert("value",index=-1)
    
    if animate:
        ## save key frames for head
        bpy.data.shape_keys['head_blink_shapekey'].key_blocks['Look_down'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['head_blink_shapekey'].key_blocks['Look_up'].keyframe_insert("value",index=-1)
        ## save key frames for upper eye lashes
        bpy.data.shape_keys['left_upper_eyelashes_blink_shapekey'].key_blocks['Look_down'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['left_upper_eyelashes_blink_shapekey'].key_blocks['Look_up'].keyframe_insert("value",index=-1)
        ## save key frames for lower eye lashes
        bpy.data.shape_keys['left_lower_eyelashes_blink_shapekey'].key_blocks['Look_down'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['left_lower_eyelashes_blink_shapekey'].key_blocks['Look_up'].keyframe_insert("value",index=-1)
        ## save key frames for eye lids
        bpy.data.shape_keys['left_eyelid_blink_shapekey'].key_blocks['Look_down'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['left_eyelid_blink_shapekey'].key_blocks['Look_up'].keyframe_insert("value",index=-1)

    # also mix skin color based on looking down
    color_mix_node = bpy.data.materials['skin_material'].node_tree.nodes['color_mix']
    color_mix_node.inputs['Fac'].default_value = max(0, 1.0-ld_amt)

    if animate:
        color_mix_node.inputs['Fac'].keyframe_insert("default_value", index=-1)

    # if it exists, blend the displacement node also
    if 'disp_mix' in bpy.data.materials['skin_material'].node_tree.nodes.keys():
        disp_mix_node = bpy.data.materials['skin_material'].node_tree.nodes['disp_mix']
        disp_mix_node.inputs['Fac'].default_value = max(0, 1.0-ld_amt)

        if animate:
            disp_mix_node.inputs['Fac'].keyframe_insert("default_value", index=-1)