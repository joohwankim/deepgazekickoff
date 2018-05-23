import bpy
import random

from mathutils import Vector
from math import *

### ----------------------------------------------------------------
### RANDOMLY CHHOOSE IRIS COLOR
### ----------------------------------------------------------------

subjs_dark_iris = ["f03", "f05", "m02", "m04"]

def randomize_iris_color(subj_id=None):

    tree = bpy.data.materials['Material_sclera'].node_tree
    nds, lks = tree.nodes, tree.links

    n_mix_iris_sclera = nds['Mix_iris_sclera']
    n_disp = nds['Disp_multiply']

    # remove link to old color
    old_link = n_mix_iris_sclera.inputs['Color2'].links[0]
    lks.remove(old_link)

    # choose new iris color node and create link
    iris_options = [
    	(nds['Texture_iris_brown'], nds['Disp_iris_brown']),
    	(nds['Texture_iris_amber'], nds['Disp_iris_amber']),
    	(nds['Texture_iris_grey'], nds['Disp_iris_grey']),
    	(nds['Texture_iris_blue'], nds['Disp_iris_blue'])]

    if subj_id:
        if subj_id in subjs_dark_iris: iris_options = iris_options[:2]

    n_random_iris_clr, n_random_iris_dsp = random.choice(iris_options)
    lks.new(n_random_iris_clr.outputs['Color'], n_mix_iris_sclera.inputs['Color2'])
    lks.new(n_random_iris_dsp.outputs['Color'], n_disp.inputs[0])

    # this action causes the preview window to refresh
    # TODO: figure out why changing the texture in nodes doesnt refresh preview
    nds['Diffuse BSDF'].inputs[1].default_value = 0


### ----------------------------------------------------------------
### SET SHAPE KEYS FOR IRIS AND PUPIL SIZE
### ----------------------------------------------------------------

def set_iris_size(scl_mesh, i_size, animated=False):

    big_amt, sml_amt = max((i_size-0.5)*2,0), max((0.5-i_size)*2,0)
    scl_mesh.shape_keys.key_blocks['Iris_big'].value = big_amt
    scl_mesh.shape_keys.key_blocks['Iris_small'].value = sml_amt

    if (animated):    
        bpy.data.shape_keys['sclera_iris_pupil_shapekey'].key_blocks['Iris_big'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['sclera_iris_pupil_shapekey'].key_blocks['Iris_small'].keyframe_insert("value",index=-1)


def set_pupil_size(scl_mesh, p_size, animated=False):

    big_amt, sml_amt = max((p_size-0.5)*2,0), max((0.5-p_size)*2,0)
    scl_mesh.shape_keys.key_blocks['Pupil_big'].value = big_amt
    scl_mesh.shape_keys.key_blocks['Pupil_small'].value = sml_amt

    if (animated):    
        bpy.data.shape_keys['sclera_iris_pupil_shapekey'].key_blocks['Pupil_big'].keyframe_insert("value",index=-1)
        bpy.data.shape_keys['sclera_iris_pupil_shapekey'].key_blocks['Pupil_small'].keyframe_insert("value",index=-1)


def reset_eye_sizes(scl_mesh):
    
    scl_mesh.shape_keys.key_blocks['Iris_big'].value = 0.5
    scl_mesh.shape_keys.key_blocks['Iris_small'].value = 0.5
    scl_mesh.shape_keys.key_blocks['Pupil_big'].value = 0.5
    scl_mesh.shape_keys.key_blocks['Pupil_small'].value = 0.5


# EXAMPLE USAGE:
#
# scl_mesh = bpy.data.meshes['Sclera']
# randomize_iris_color()
# set_iris_size(scl_mesh, random.gauss(0.5,0.1)) 
# set_pupil_size(scl_mesh, random.gauss(0.5,0.1))


### ----------------------------------------------------------------
### GAZE ESTIMATION
### ----------------------------------------------------------------

def get_look_vec_cam_coords(eye, cam):
    
    eye_rot_mat = eye.matrix_world.to_quaternion().to_matrix()
    look_vec = eye_rot_mat * Vector((0,0,1))
    eye_pos_for_cam = cam.matrix_world.inverted() * eye.location
    look_pt_for_cam = cam.matrix_world.inverted() * (eye.location+look_vec)
    
    return look_pt_for_cam - eye_pos_for_cam


def get_pitch(eye):

    eye_rot_mat = eye.matrix_world.to_quaternion().to_matrix()
    look_vec = eye_rot_mat * Vector((0,1,0))
    x,y,z = look_vec.to_tuple()
    #return degrees(atan(z/sqrt(x**2 + y**2)))
    return degrees(atan(z/y))


def get_yaw(eye):

    eye_rot_mat = eye.matrix_world.to_quaternion().to_matrix()
    look_vec = eye_rot_mat * Vector((0,1,0))
    x,y,z = look_vec.to_tuple()
    return degrees(atan((x/y)))
