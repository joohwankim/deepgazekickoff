from mathutils import Vector, Matrix, Euler
from math import sin, cos, radians, tan
import numpy as np
from random import uniform, gauss
import math, pdb

# blender imports
import bpy
import bpy_extras

# project imports
import eyelids
import eye
#import env
#import utils
import imp
imp.reload(eyelids)
imp.reload(eye)
#imp.reload(env)
#imp.reload(utils)

from eyelids import *
from eye import *
#from env import *
#from utils import *

###################################
# GEOMETRICAL CONFIGURATION
###################################
def load_geometrical_configuration(model_name):
    centralEyeLocation = Vector((-0.625579, 1.21064, 3.07758))  # The eye position assumed by the HMD. In reality this does not match with the eye position due to the slippage.
    eyeLookAtOrigin = centralEyeLocation + Vector((0.31, 0.0, 0.0))
    default_setting_models = [
        'nxp_female_01','nxp_female_02','nxp_female_03','nxp_female_04','nxp_female_05',
        'nxp_male_01','nxp_male_02','nxp_male_03','nxp_male_04','nxp_male_05',
    ]
    # nxp models
    if model_name in default_setting_models:
        headEuler = Euler((0.0, 0.0, 0.0),'XYZ')
        geo = {
            'headEuler':headEuler,
            'headRot':Matrix.Rotation(headEuler[0],3,'X'),
            'headUp':Vector((0,0,1)) * Matrix.Rotation(headEuler[0],3,'X'),
            'eyeLookAtOrigin':eyeLookAtOrigin,
            'centralEyeLocation':centralEyeLocation,
            'centralHeadLocation':centralEyeLocation + Vector((0.0, 0.13, 0.0)), # assumes upright configuration
        }
    # non-nxp models. In these cases we pass variable necessary to convert them to NXP models.
    # The scaling values are estimated by Michael Stengel and is described in our Confluence page (Documentation / In-house Blender model)
    elif model_name == 'female_01': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.82 }
    elif model_name == 'female_02': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.93 }
    elif model_name == 'female_03': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.84 }
    elif model_name == 'female_04': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.86 }
    elif model_name == 'female_05': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.85 }
    elif model_name == 'male_01': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.96 }
    elif model_name == 'male_02': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.79 }
    elif model_name == 'male_03': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.90 }
    elif model_name == 'male_04': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.88 }
    elif model_name == 'male_05': # default case where geometrical setup is not completely determined.
        geo = { 'centralEyeLocation':centralEyeLocation, 'headScaling':0.84 }
    return geo

###################################
# FUNCTIONS FOR EYE ANIMATION
###################################

# get handles to the various Blender objects
# cam = bpy.data.objects['Cam_eye']
# eye_obj = bpy.data.objects["left_eye"]
# eyelashes_lower_obj = bpy.data.objects["left_lower_eyelashes"]
# eyelashes_upper_obj = bpy.data.objects["left_upper_eyelashes"]
# head_obj = bpy.data.objects["head"]
# scl_obj = bpy.data.objects["left_sclera"]
# eye_LookAt = bpy.data.objects['left_eye_lookat']
# cam_LookAt = bpy.data.objects['Cam_eye_lookAt']
# scl_mesh = bpy.data.meshes['sclera_mesh']

# D, C = bpy.data, bpy.context

def clear_animation():
    # reset objects to their inital state
    # clear animation frames
    bpy.data.objects['left_eye_lookat'].animation_data_clear()
    bpy.data.objects["left_eye"].animation_data_clear()
    bpy.data.objects["head"].animation_data_clear()

def look_at_mat(target, up, pos):
    # zaxis = (target - pos).normalized()
    # xaxis = up.cross(zaxis).normalized()
    # yaxis = zaxis.cross(xaxis).normalized()
    yaxis = (target - pos).normalized()
    xaxis = yaxis.cross(up).normalized()
    zaxis = xaxis.cross(yaxis).normalized()
    return Matrix([xaxis, yaxis, zaxis]).transposed()

def look_at(eye_obj, target, geo):
    # apply the effect of rotation of the parent node.
    # There should be a way to get the absolute position of an object when it is a child.
    # Right now I couldn't figure out how, so I am calculating it from the relative distance of the eye center to head center when head is upright.
    # This should be updated for any new face model.
    #pos = obj.location
    pos = eye_obj.parent.location + geo['headRot'] * (geo['centralEyeLocation'] - geo['centralHeadLocation']) # position of the eye
    obj_rot_mat = geo['headRot'].inverted() * look_at_mat(target,geo['headUp'],pos)
    
    #if obj.parent:
    #    P = obj.parent.matrix.decompose()[1].to_matrix()
    #    obj_rot_mat = P * obj_rot_mat * P.inverted()
        
    eye_obj.rotation_mode = 'XYZ'
    eye_obj.rotation_euler = obj_rot_mat.to_euler('XYZ')
#    pdb.set_trace()

def resetEye(eye_handle):
    eye_handle.rotation_euler[0] = radians(0)
    eye_handle.rotation_euler[1] = 0
    eye_handle.rotation_euler[2] = radians(0)

def animateEye(eye_handle, eyelookat_handle, geo, keyframe_setting):
    gaze = (keyframe_setting['gaze_x'],keyframe_setting['gaze_y'])
    eyeOpenness = 0.5
    if 'blink' in keyframe_setting:
        eyeOpenness = keyframe_setting['blink']
    X = radians(gaze[0])
    Y = radians(gaze[1])
    displayVirtualDistance = 5.3 # Virtual display at 53cm away
    x_target = displayVirtualDistance * np.tan(X)
    z_target = displayVirtualDistance * np.tan(Y)
    eyelookat_handle.location = geo['eyeLookAtOrigin'] + Vector((x_target, displayVirtualDistance, z_target))
    
    look_at(eye_handle,
        eyelookat_handle.location,
        geo)

    eyelookat_handle.keyframe_insert(data_path="location", index=-1)
    eye_handle.keyframe_insert(data_path="rotation_euler", index=-1)
    
    e_p, e_y = get_pitch(eye_handle), get_yaw(eye_handle)
    # pose eye lids and eye lashes
    pose_eyelids_animated(e_p, eyeOpenness, False, True)
    # intensity adjustment
    if 'iris_intensity' in keyframe_setting:
        bpy.data.materials['sclera_material'].node_tree.nodes['mult_intensity_iris'].inputs[1].default_value = keyframe_setting['iris_intensity']
        bpy.data.materials['sclera_material'].node_tree.nodes['mult_intensity_iris'].inputs[1].keyframe_insert(data_path = 'default_value', index = -1)
    if 'skin_intensity' in keyframe_setting:
        bpy.data.materials['skin_material'].node_tree.nodes['mult_intensity_look_down'].inputs[1].default_value = keyframe_setting['skin_intensity']
        bpy.data.materials['skin_material'].node_tree.nodes['mult_intensity'].inputs[1].default_value = keyframe_setting['skin_intensity']
        bpy.data.materials['skin_material'].node_tree.nodes['mult_intensity_look_down'].inputs[1].keyframe_insert(data_path = 'default_value', index = -1)
        bpy.data.materials['skin_material'].node_tree.nodes['mult_intensity'].inputs[1].keyframe_insert(data_path = 'default_value', index = -1)
    if 'sclera_intensity' in keyframe_setting:
        bpy.data.materials['sclera_material'].node_tree.nodes['mult_intensity_sclera'].inputs[1].default_value = keyframe_setting['sclera_intensity']
        bpy.data.materials['sclera_material'].node_tree.nodes['mult_intensity_sclera'].inputs[1].keyframe_insert(data_path = 'default_value', index = -1)
    bpy.context.scene.update()

def animate_keyframe(frameNumber, geo, keyframe_setting):
    bpy.context.scene.frame_set(frameNumber)
    newHeadLocation = geo['centralHeadLocation'].copy()
    if 'slippage_x' in keyframe_setting:
        newHeadLocation[0] = newHeadLocation[0] + keyframe_setting['slippage_x'] / 100.0
    if 'slippage_y' in keyframe_setting:
        newHeadLocation[1] = newHeadLocation[1] + keyframe_setting['slippage_y'] / 100.0
    if 'slippage_z' in keyframe_setting:
        newHeadLocation[2] = newHeadLocation[2] + keyframe_setting['slippage_z'] / 100.0
    bpy.data.objects["head"].location = newHeadLocation
    bpy.data.objects["head"].rotation_euler = geo['headEuler']
    bpy.data.objects["head"].keyframe_insert(data_path="location", index=-1)
    # set animated eye defined by gaze and eye openness
    animateEye(bpy.data.objects["left_eye"],bpy.data.objects['left_eye_lookat'],geo,keyframe_setting)
    irisSize = 0.5
    set_iris_size(bpy.data.meshes['sclera_mesh'], irisSize, True)
    pupilSize = 0.5
    if 'pupil_size' in keyframe_setting:
        pupilSize = keyframe_setting['pupil_size']
    set_pupil_size(bpy.data.meshes['sclera_mesh'], pupilSize, True)
