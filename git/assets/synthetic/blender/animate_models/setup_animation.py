# system imports
from mathutils import Vector, Matrix, Euler
from math import sin, cos, radians, tan
import os
import numpy as np
import pickle
from random import uniform, gauss
import math, itertools
import pdb

# project imports
import imp
import animateNXPeye
imp.reload(animateNXPeye)
from animateNXPeye import *

def manipulate_led_strength(instruction):
    num_leds = 26
    # first, turn off all the leds
    led_count = 26
    led_obs = list()
    for ii in range(led_count):
        if ii+1 < 10:
            exec("led_obs.append(bpy.data.objects['led_00%s'])"%str(ii+1))
        else:
            exec("led_obs.append(bpy.data.objects['led_0%s'])"%str(ii+1))
    for led in led_obs:
        led.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 0.0
    # determine led strength according to the instruction
    if instruction == 'as_in_prototype':
        led_on_indices = np.asarray([1, 6, 16, 19])
        led_strengths = np.ones(led_on_indices.shape) * 0.5
    elif instruction == 'random_on_constant_strength':
        led_on_indices = np.arange(1, 27, 1)
        led_strengths = np.round(np.random.random((26,)))
        led_strengths = 2 * led_strengths / np.sum(led_strengths) # making total intensity across all leds 2
    elif instruction == 'random_on_random_strength':
        led_on_indices = np.arange(1, 27, 1)
        led_strengths = np.random.random((26,))
        led_strengths = 2 * led_strengths / np.sum(led_strengths) # making total intensity across all leds 2
    # actually turn on / off leds
    for ii, strength in zip(led_on_indices, led_strengths):
        exec("bpy.data.objects['led_%s'].data.node_tree.nodes['Emission'].inputs['Strength'].default_value = %s"%(str(ii).zfill(3), str(strength)))
        

###################################
# CREATE EYE ANIMATION
###################################
def setup_animation(model_path, plan, output_path):
    # IMPORTANT: The filename should match the model name because it is inferred out of the file name.
    # plan:
    # information on changes on variables per each key frame.
    # output_path (assumes a .blend extension):
    # The file path based on which the blend file and csv will be generated.
    bpy.ops.wm.open_mainfile(filepath=model_path)
    clear_animation()
    bpy.data.scenes["Scene"].cycles.samples = 1000
    model_name = os.path.basename(bpy.data.filepath).replace('.blend','').replace('_withouthead','')
    geo = load_geometrical_configuration(model_name)
    
    # Manipulate LEDs and remove the LED-related keys (because these parameters don't have the consistent formats like others).
    led_strength_instruction = 'as_in_prototype'
    if 'led_strength_instruction' in plan:
        led_strength_instruction = plan['led_strength_instruction']
        del plan['led_strength_instruction']
    manipulate_led_strength(led_strength_instruction)
    if 'show_glint' in plan:
        if plan['show_glint']: # connect transmission ray in 'Light Path' node box to 2nd input in 'Math' node box. This creates glint on the cornea.
            mat = bpy.data.materials['cornea_material']
            mat.node_tree.links.new(mat.node_tree.nodes['Light Path'].outputs[6], mat.node_tree.nodes['Math'].inputs[1])
        else: # connect reflection ray in 'Light Path' node box to 2nd input in 'Math' node box. This ignores glint.
            mat = bpy.data.materials['cornea_material']
            mat.node_tree.links.new(mat.node_tree.nodes['Light Path'].outputs[5], mat.node_tree.nodes['Math'].inputs[1])
        del plan['show_glint']

    samples = dict() # contains stratified sample points for each parameter
    num_totalSamples = 1
    for key, value in plan.items():
        value['interval'] = (value['range'][1] - value['range'][0]) / value['num_intervals']
        samples[key] = np.linspace(value['range'][0] + value['interval']/2, value['range'][1] - value['interval']/2, num = value['num_intervals']) # stratified sampling of each dimension
        num_totalSamples *= value['num_intervals']

    print("Total number of samples: %d\n"%(2 * num_totalSamples))

    #logfile = open('C:/Work/DeepLearning/deep-gaze/code/dataset/syntheyes/scripts/fileName.csv', 'w')
    log_filename = output_path.replace('.blend','.csv')
    logfile = open(log_filename, 'w')
    log_msg = 'EYE,ORIGINAL_EYE_X,ORIGINAL_EYE_Y,ORIGINAL_EYE_Z,'
    sorted_par_list = list(plan.keys())
    sorted_par_list.sort()
    for key in sorted_par_list:
        log_msg = log_msg + key.upper() + ','
    logfile.write(log_msg + '\n')

    frameNumber = 0
    log_msg = ''
    for x in itertools.product(*samples.values()):
        if np.mod(frameNumber,1000) == 0:
            print(frameNumber)
        left_eye_keyframe_setting = dict(zip(samples.keys(),x)) # dictionary containing one set of parameters, which is the product of regular stratified sample points
        right_eye_keyframe_setting = dict(zip(samples.keys(),x)) # dictionary containing one set of parameters, which is the product of regular stratified sample points
        for key in left_eye_keyframe_setting.keys():
            if plan[key]['jitter']:
                left_eye_keyframe_setting[key] = left_eye_keyframe_setting[key] + plan[key]['interval'] * (np.random.rand(1) - 0.5) # random jitter
            if key == 'gaze_x' or key == 'slippage_x':
                right_eye_keyframe_setting[key] = - left_eye_keyframe_setting[key]
            else:
                right_eye_keyframe_setting[key] = left_eye_keyframe_setting[key]
                    
        # animate and create keyframe for the left eye
        animate_keyframe(frameNumber, geo, left_eye_keyframe_setting)
        log_msg = log_msg + 'L,%.8f,%.8f,%.8f,'%(geo['centralEyeLocation'][0]*100.0,geo['centralEyeLocation'][1]*100.0,geo['centralEyeLocation'][2]*100.0)
        for key in sorted_par_list:
            log_msg = log_msg + '%.8f,'%left_eye_keyframe_setting[key]
        log_msg = log_msg + '\n'
        frameNumber += 1
        
        # animate and create keyframe for the right eye
        animate_keyframe(frameNumber, geo, right_eye_keyframe_setting)
        log_msg = log_msg + 'R,%.8f,%.8f,%.8f,'%(geo['centralEyeLocation'][0]*100.0 + 62.0,geo['centralEyeLocation'][1]*100.0,geo['centralEyeLocation'][2]*100.0)

        for key in sorted_par_list:
            log_msg = log_msg + '%.8f,'%right_eye_keyframe_setting[key]
        log_msg = log_msg + '\n'
        
        frameNumber += 1
    #    pdb.set_trace()

    logfile.write(log_msg)
    logfile.close()

    print("\n-----------------script complete-----------------")
    bpy.ops.wm.save_as_mainfile(filepath = output_path)
