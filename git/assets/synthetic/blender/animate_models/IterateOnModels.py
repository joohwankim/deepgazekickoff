import os, pdb
from shutil import copyfile
import bpy

# Add any necessary files below.
necessary_files = [
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/animateNXPeye.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/eye.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/eyelids.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/NXPModelPrepHelper.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/NXPSyntheticEyeDataPlan.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/render_with_cuda.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/setup_animation.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/setup_object_masks.py',
    'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/setup_rendering.py',
]
    #'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye/scripts/updateNXPeye.py', # this doesn't work.. I could not resolve the context issue.

for necessary_file in necessary_files:
    if os.path.basename(necessary_file) not in bpy.data.texts.keys():
        bpy.ops.text.open(filepath = necessary_file)

# project imports
import imp
import setup_animation
imp.reload(setup_animation)
from setup_animation import *
import NXPSyntheticEyeDataPlan
imp.reload(NXPSyntheticEyeDataPlan)
from NXPSyntheticEyeDataPlan import *
import setup_object_masks
imp.reload(setup_object_masks)
from setup_object_masks import *
#import updateNXPeye
#imp.reload(updateNXPeye)
#from updateNXPeye import *

###################################
# CHOOSE PLAN
###################################

# available plans: 'IrisIntensityTest', 'ScleraIntensityTest', 'SkinIntensityTest', 'Plan001', 'SimplePlan'

###################################
# SETUP
###################################

# in case a pre-loaded plan needs to be used...
original_path = bpy.data.filepath
# plan_names = ['TestRobustnessToBlink','TestRobustnessToPupilSize','TestRobustnessToTransversalSlippage','TestRobustnessToDepthSlippage','TestRobustnessToSlippage',] # do [''] for when making modifications on models
plan_names = ['Plan002','Plan002_no_glint'] # do [''] for when making modifications on models
model_folder = 'X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scenes'
# NOTE: The following line should match the folder path in the rendering environment (e.g. currently I am using NFS share mounted on a docker container running on SaturnV).
blender_output_folder_local = 'X:/scratch/sckim/blender_output'
blender_output_folder_dgx = '/media/scratch/sckim/blender_output'
if not os.path.exists(blender_output_folder_local):
    os.mkdir(blender_output_folder_local)

## All models
#model_names = [
#                  'nxp_female_02','nxp_female_01','nxp_female_04','nxp_female_05',
#                  'nxp_male_01','nxp_male_03','nxp_male_04','nxp_male_05',
#                  'nxp_female_03','nxp_male_02',
#              ]
# All models except for nxp_female_01
#model_names = [
#                  'nxp_female_02','nxp_female_01',
#                  'nxp_female_04','nxp_female_05',
#                  'nxp_male_01','nxp_male_03','nxp_male_04','nxp_male_05',
#                  'nxp_female_03','nxp_male_02',
#              ]
model_names = [
                  'nxp_female_03',
                  #'nxp_male_01','nxp_male_03','nxp_male_04','nxp_male_05','nxp_male_02',
              ]

## Quick models only
#model_names = [
#                  'nxp_female_01','nxp_female_02','nxp_female_04','nxp_female_05',
#                  'nxp_male_01','nxp_male_03','nxp_male_04','nxp_male_05',
#              ]

## One model, for debugging.
#model_names = [
#                  'nxp_female_01'
#              ]

## remaining models
#model_names = [
#                  'nxp_female_01','nxp_female_02','nxp_female_04','nxp_female_05',
#                  'nxp_male_01','nxp_male_03','nxp_male_04','nxp_male_05',
#                  'nxp_female_03','nxp_male_02',
#              ]


###################################
# GENERATE ANIMATION
###################################

for model_name in model_names:
    for plan_name in plan_names:
        # set up animation.
        model_path = os.path.join(model_folder, model_name, 'scenes', model_name + '.blend')
        if len(plan_name) > 0: # Set up animation.
            animated_blendfile_path = os.path.join(model_path.replace('.blend','') + '_' + plan_name + '.blend')
            plan = loadPlan(plan_name)
            setup_animation(model_path, plan, animated_blendfile_path)
            
            # set up object mask generation.
            # The render output path should be described in the context of the rendering environment. Currently it is NFS share mounted on the docker container running on SaturnV.
            render_output_path_dgx = os.path.join(blender_output_folder_dgx, os.path.basename(animated_blendfile_path))
            setup_object_masks(animated_blendfile_path, render_output_path_dgx)
            
            # also create the render output folder. Describe it in local os 'language' so that the current os can create it.
            render_output_path_local = os.path.join(blender_output_folder_local, os.path.basename(animated_blendfile_path).replace('.blend',''))
            if not os.path.exists(render_output_path_local):
                os.mkdir(render_output_path_local)
                
            # copy the csv file to the output folder.
            src_csv_filepath = animated_blendfile_path.replace('.blend','.csv')
            dst_csv_filepath = os.path.join(render_output_path_local, os.path.basename(animated_blendfile_path).replace('.blend','.csv'))
            copyfile(src_csv_filepath,dst_csv_filepath)

        else: # No animation plan. This is for modifying the reference model. Iterate the following on the model files.
            updateNXPeye(model_path)
            
        
bpy.ops.wm.open_mainfile(filepath = original_path) # come back to the original file