import os, pdb
from mathutils import Matrix, Vector

import imp
import NXPModelPrepHelper
imp.reload(NXPModelPrepHelper)
from NXPModelPrepHelper import *
import animateNXPeye
imp.reload(animateNXPeye)
from animateNXPeye import *

def updateNXPeye(model_path):
    # commenting out because any accidental call to this code can alter the content of the reference model.
#    bpy.ops.wm.open_mainfile(filepath=model_path)
#    bpy.data.objects['left_eye'].parent = None
#    delete_hierarchy(bpy.data.objects['left_eye'])

#    bpy.ops.wm.save_as_mainfile(filepath = model_path.replace('.blend') + '_test.blend')
#    bpy.ops.wm.open_mainfile(filepath=model_path.replace('.blend') + '_test.blend')
#    geo = load_geometrical_configuration(os.path.basename(model_path).replace('.blend',''))
#    blendfilepath = 'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye/scenes/eye/NXP_eye.blend'
#    section = '\\Object\\'
#    objects = ['left_eye','left_cornea','left_sclera']

#    # overriding context for object importing.
#    for window in bpy.context.window_manager.windows:
#        screen = window.screen

#        for area in screen.areas:
#            if area.type == 'VIEW_3D':
#                override = {'window': window, 'screen': screen, 'area': area}
#                bpy.ops.screen.screen_full_area(override)
#                break
#    bpy.ops.object.group_instance_add('INVOKE_DEFAULT')

#    for obj in objects:
#        bpy.ops.wm.append(filename = obj, directory = blendfilepath + section)
#    bpy.data.objects['left_eye'].location = Vector((0.0,0.0,0.0))
#    identity_matrix = Matrix()
#    Matrix.identity(identity_matrix)
#    bpy.data.objects['left_eye'].matrix_world =  identity_matrix
#    parent(bpy.data.objects['left_sclera'], bpy.data.objects['left_eye'])
#    parent(bpy.data.objects['left_cornea'], bpy.data.objects['left_eye'])
#    bpy.data.objects['left_eye'].location = geo['centralEyeLocation']
#    parent(bpy.data.objects['left_eye'], bpy.data.objects['head'])
#    bpy.ops.screen.back_to_previous() # exit full screen mode before closing.
#    bpy.ops.wm.save_as_mainfile(filepath = model_path.replace('.blend') + '_test.blend')
