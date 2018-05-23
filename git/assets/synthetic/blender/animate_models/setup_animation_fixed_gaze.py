# system imports
from mathutils import Vector, Matrix, Euler
from math import sin, cos, radians, tan
import os
import numpy as np
import pickle
from random import uniform, gauss
import math
import pdb


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

import spiral

###################################
# SETUP
###################################

gazeInterval        = 10     # in degrees
horGazeLimit        = 10    # [-ey,+ey] - in degrees
vertGazeLimit       = 10    # [-ey,+ey] - in degrees
gazeJitter          = True

numPupilStates      = 10     # number
minPupilSize        = 0     # in range [0,20]
maxPupilSize        = 1    # in range [0,20]

numEyelidStates     = 10     # number
minEyelidOpenness   = 0     # in range [0,20]
maxEyelidOpenness   = 1    # in range [0,20]

simulateBlink       = True

horSlippagePoints = 3
vertSlippagePoints = 3
horSlippageInterval = 3
vertSlippageInterval = 3

depthShiftStates = 3     # number
minDepthShift    = 0     # in range [0,15] mm
maxDepthShift    = 15    # in range [0,15] mm

#gazeInterval        = 4     # in degrees
#horGazeLimit        = 20    # [-ey,+ey] - in degrees
#vertGazeLimit       = 16    # [-ey,+ey] - in degrees
#gazeJitter          = True

#numPupilStates      = 10     # number
#minPupilSize        = 0     # in range [0,20]
#maxPupilSize        = 1    # in range [0,20]

#numEyelidStates     = 10     # number
#minEyelidOpenness   = 0     # in range [0,20]
#maxEyelidOpenness   = 1    # in range [0,20]

#simulateBlink       = True

#horSlippagePoints = 3
#vertSlippagePoints = 3
#horSlippageInterval = 3
#vertSlippageInterval = 3


#depthShiftStates = 1     # number
#minDepthShift    = 0     # in range [0,15] mm
#maxDepthShift    = 0    # in range [0,15] mm


###################################
# GEOMETRICAL CONFIGURATION
###################################
eyeLookAtOrigin = Vector((-0.616346, 1.16598, 3.0678)) # The eye position assumed by the HMD. In reality this does not match with the eye position due to the slippage.
headLocation = Vector((-0.616346, 1.29598, 3.0678)) # for the rescaled head of female_01
#headEuler = Euler((radians(50.0), 0.0, 0.0),'XYZ')
headEuler = Euler((radians(0.0), 0.0, 0.0),'XYZ')
headRot = Matrix.Rotation(headEuler[0],3,'X')
headUp = Vector((0,0,1)) * headRot
uprightEyeLocation = Vector((-0.616346, 1.16598, 3.0678))
uprightHeadToEye = uprightEyeLocation - headLocation # Don't know how to get the absolute position of an object when it's a child. So I calculate it from eye's location when head is upright.

print("\n-----------------\nCreate eye animation\n-----------------\n")

def generate1DStratifiedSamples(min0, max0, num0):
    # distribute points
    min1 = min0 + (max0 - min0) / num0 / 2.0
    max1 = max0 - (max0 - min0) / num0 / 2.0
    points = np.linspace(min1, max1, num0)
    return points

def generate2DStratifiedSamples(min0, max0, interval0, min1, max1, interval1):

    num0 = (max0 - min0) / interval0
    num1 = (max1 - min1) / interval1
    
    gx = generate1DStratifiedSamples(min0,max0,num0)
    print(gx)
    gy = generate1DStratifiedSamples(min1,max1,num1)
    print(gx)
    
    points = np.ndarray([len(gx)*len(gy),2])
    
    sampleidx = 0
    for px in gx:
        for py in gy:
            points[sampleidx] = [px, py]
            sampleidx = sampleidx + 1
    return points

def generate3DStratifiedSamples(min0, max0, interval0, min1, max1, interval1, min2, max2, interval2):

    num0 = (max0 - min0) / interval0
    num1 = (max1 - min1) / interval1
    num2 = (max2 - min2) / interval2
    
    gx = generate1DStratifiedSamples(min0,max0,num0)
    gy = generate1DStratifiedSamples(min1,max1,num1)
    gz = generate1DStratifiedSamples(min2,max2,num2)
    
    points = np.ndarray([len(gx)*len(gy)*len(gz),2])
    
    sampleidx = 0
    for px in gx:
        for py in gy:
            for pz in gz:
                points[sampleidx] = [px, py, pz]
                sampleidx = sampleidx + 1
    return points

def randomSampleStratified1D(origSample, interval):
    return origSample + np.random.uniform(0,interval)

def randomSampleStratified2D(origSample, interval0, interval1):
    sample = np.ndarray([2,1])
    sample[0] = origSample[0] + + np.random.uniform(0,interval0)
    sample[1] = origSample[1] + + np.random.uniform(0,interval1)
    return sample

def randomSampleStratified3D(origSample, interval0, interval1, interval2):
    sample = np.ndarray([3,1])
    sample[0] = origSample[0] + + np.random.uniform(0,interval0)
    sample[1] = origSample[1] + + np.random.uniform(0,interval1)
    sample[2] = origSample[2] + + np.random.uniform(0,interval2)
    return sample

# GAZE POINTS

gazePoints = generate2DStratifiedSamples(-horGazeLimit,horGazeLimit, gazeInterval, -vertGazeLimit, vertGazeLimit, gazeInterval)
numGazePoints = len(gazePoints)
#print(gazePoints)
#print(gazePoints[-1])
#print(randomSampleStratified2D(gazePoints[-1], gazeInterval,gazeInterval))
#print(gazePoints)

# validate gaze positions ?

# SLIPPAGE

#slippagePoints = generate2DStratifiedSamples(-horSlippageLimit,horSlippageLimit, slippageInterval, -vertSlippageLimit, vertSlippageLimit, slippageInterval)
#numSlippagePoints = len(slippagePoints)

#print(slippagePoints)
#print(slippagePoints[-1])
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))
#print(randomSampleStratified2D(slippagePoints[-1], slippageInterval,slippageInterval))

# compute slippage motion
print(vertSlippagePoints)
print(horSlippagePoints)

gridPoints = np.zeros([vertSlippagePoints,horSlippagePoints])
slippageIndices = spiral.computeSpiralMotion(gridPoints)

print(slippageIndices)
numSlippagePoints = len(slippageIndices)

# depth shift

depthShiftPoints = generate1DStratifiedSamples(minDepthShift, maxDepthShift, depthShiftStates)
numDepthShiftPoints = len(depthShiftPoints)
depthShiftInterval = (maxDepthShift-minDepthShift)/numDepthShiftPoints
print(depthShiftPoints)



# pupil adjustment points

pupilAdjustmentPoints = generate1DStratifiedSamples(minPupilSize, maxPupilSize, numPupilStates)
numPupilAdjustmentPoints = len(pupilAdjustmentPoints)
pupilAdjInterval = (maxPupilSize-minPupilSize)/numPupilStates
print(pupilAdjustmentPoints)
#print(pupilAdjustmentPoints[0])
#print(randomSampleStratified1D(pupilAdjustmentPoints[0], pupilAdjInterval))

# eye openness points

eyeOpennessPoints = generate1DStratifiedSamples(minEyelidOpenness, maxEyelidOpenness, numEyelidStates)
numEyeOpennessPoints = len(eyeOpennessPoints)
eyeOpennessInterval = (maxEyelidOpenness-minEyelidOpenness)/numEyelidStates

if ( simulateBlink == False ):
    numEyeOpennessPoints = 1

#print(eyeOpennessPoints)
#print(eyeOpennessPoints[-1])
#print(randomSampleStratified1D(eyeOpennessPoints[-1], eyeOpennessInterval))


# number of samples

totalNumSamples = numGazePoints * numSlippagePoints * numPupilAdjustmentPoints * numEyeOpennessPoints * numDepthShiftPoints

print("number of gaze points : %d"%numGazePoints)
print("number of slippage points : %d"%numSlippagePoints)
print("number of depth shift points : %d"%numDepthShiftPoints)
print("number of pupil adjustment points : %d"%numPupilAdjustmentPoints)
print("number of eye openness points : %d"%numEyeOpennessPoints)

print("total number of samples: %d"%totalNumSamples)


# get handles to the various Blender objects
cam = bpy.data.objects['Cam_eye']
eye_obj = bpy.data.objects["EyeLeft"]
eyelashes_lower_obj = bpy.data.objects["EyeLeft_Eyelashes_lower"]
eyelashes_upper_obj = bpy.data.objects["EyeLeft_Eyelashes_upper"]
head_obj = bpy.data.objects["Head"]
scl_obj = bpy.data.objects["ScleraLeft"]
eye_LookAt = bpy.data.objects['EyeLeft_lookAt']
cam_LookAt = bpy.data.objects['Cam_eye_lookAt']
scl_mesh = bpy.data.meshes['Sclera']

D, C = bpy.data, bpy.context

###################################
# CAMERA ORIENTATION AND POSITION
###################################

#C.scene.update()

###################################
# LED ORIENTATION AND POSITION
###################################

#C.scene.update()

# turn off the environment lighting:
#old_env_node = removeEnvLighting()

###################################
# CREATE EYE ANIMATION
###################################

def look_at_mat(target, up, pos):
    # zaxis = (target - pos).normalized()
    # xaxis = up.cross(zaxis).normalized()
    # yaxis = zaxis.cross(xaxis).normalized()
    yaxis = (target - pos).normalized()
    xaxis = yaxis.cross(up).normalized()
    zaxis = xaxis.cross(yaxis).normalized()
    return Matrix([xaxis, yaxis, zaxis]).transposed()

def look_at(obj, target, up, initPoseMat):
    # apply the effect of rotation of the parent node.
    # There should be a way to get the absolute position of an object when it is a child.
    # Right now I couldn't figure out how, so I am calculating it from the relative distance of the eye center to head center when head is upright.
    # This should be updated for any new face model.
    #pos = obj.location
    pos = obj.parent.location + initPoseMat * uprightHeadToEye # position of the eye
    obj_rot_mat = initPoseMat.inverted() * look_at_mat(target,up,pos)
    
    #if obj.parent:
    #    P = obj.parent.matrix.decompose()[1].to_matrix()
    #    obj_rot_mat = P * obj_rot_mat * P.inverted()
        
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = obj_rot_mat.to_euler('XYZ')
#    pdb.set_trace()

def resetEye(eye_handle):
    eye_handle.rotation_euler[0] = radians(0)
    eye_handle.rotation_euler[1] = 0
    eye_handle.rotation_euler[2] = radians(0)

def animateEye(eye_handle, eyelookat_handle, gaze, eyeOpenness):

    X = radians(gaze[0])
    Y = radians(gaze[1])
    x_target = 1 * np.tan(X)
    z_target = 1 * np.tan(Y)
    eyelookat_handle.location = eyeLookAtOrigin + Vector((x_target, 1, z_target))
    
    look_at(eye_handle,
        eyelookat_handle.location,
        headUp,
        headRot)

    eye_LookAt.keyframe_insert(data_path="location", index=-1)
    eye_obj.keyframe_insert(data_path="rotation_euler", index=-1)
    
    C.scene.update()
    
    e_p, e_y = get_pitch(eye_handle), get_yaw(eye_handle)

    # pose eye lids and eye lashes
    pose_eyelids_animated(e_p, eyeOpenness, True, True)
    
    # check if animation is correct
    #print("target : %.2f   %.2f"%(gaze[0], gaze[1]))
    #print("Eye pitch: %f" % (e_p))
    #print("Eye yaw: %f" % (e_y))


# reset objects to their inital state
# clear animation frames
eye_LookAt.animation_data_clear()
eye_obj.animation_data_clear()
head_obj.animation_data_clear()

#head_obj.location = Vector((0.05, 0.0, -0.03))

#bpy.data.shape_keys['Key'].key_blocks['Look_down'].animation_data_clear()
#bpy.data.shape_keys['Key'].key_blocks['Look_up'].animation_data_clear()

# reset eye orientation
resetEye(eye_obj)

# ANIMATION

# initialize frame number
frameNumber = 0


#logfile = open('C:/Work/DeepLearning/deep-gaze/code/dataset/syntheyes/scripts/fileName.csv', 'w')
filename = "%s%s" %(bpy.data.filepath, '.csv')
logfile = open(filename, 'w')
logfile.write('EYE, GAZE_DEGREE_X, GAZE_DEGREE_Y, SLIPPAGE_SHIFT_X, SLIPPAGE_SHIFT_Y, PUPILSIZE, IRISSIZE, EYEOPENNESS, DEPTH\n')
log_msg = []

# animate for every gaze point
for pupilAdjustmentIdx in range(numPupilAdjustmentPoints):
    for eyeOpennessIdx in range(numEyeOpennessPoints):
        for depthShiftIdx in range(numDepthShiftPoints):

            # ----------------------------
            # DEPTH SHIFT SIMULATION
            # ----------------------------
            
            #depth = randomSampleStratified1D(depthShiftPoints[depthShiftIdx], depthShiftInterval) # stratified randomized
            depth = 0.0 - (depthShiftPoints[depthShiftIdx] / 100.0) # mm to project units - head offset

            for slippageIdx in range(numSlippagePoints):
                #print("slippageIdx %d"%slippageIdx)
            
                for gazePointIdx in range(numGazePoints):
                    # Gaze point in the outermost loop. We vary other variables while keeping gaze point the same
                    # get randomized gaze    
                    
                    #gaze = gazePoints[gazePointIdx]
                    if gazeJitter:
                        gaze = randomSampleStratified2D(gazePoints[gazePointIdx], gazeInterval,gazeInterval)
                    else:
                        gaze = gazePoints[gazePointIdx]
                    #gaze[0] = gaze[1] = 0
                
                    print(frameNumber)
                
                    # go to frame to be animated
                    C.scene.frame_set(frameNumber)

                    # ----------------------------
                    # SLIPPAGE SIMULATION
                    # ----------------------------

                    head_obj.location = headLocation
                    head_obj.rotation_euler = headEuler
                    
                    # generate slippage value
                    centralX = int(horSlippagePoints / 2)
                    centralY = int(vertSlippagePoints / 2)

                    # static
                    #shiftX = (slippageIndices[slippageIdx][0] - centralX) * horSlippageInterval
                    #shiftY = (slippageIndices[slippageIdx][1] - centralY) * vertSlippageInterval
                    # random shift
                    shiftX = (slippageIndices[slippageIdx][0] - centralX) * horSlippageInterval + (np.random.uniform(-0.5,0.5) * horSlippageInterval)
                    shiftY = (slippageIndices[slippageIdx][1] - centralY) * vertSlippageInterval + (np.random.uniform(-0.5,0.5) * vertSlippageInterval)

                    #print('slippage shift : x = %.2f, y = %.2f'%(shiftX,shiftY))
                    
                    head_obj.location = head_obj.location + Vector((shiftX / 100, depth, shiftY / 100)) # mm to project units ( 10 cm )
                    head_obj.keyframe_insert(data_path="location", index=-1)

                    # ----------------------------
                    # EYE OPENNESS
                    # ----------------------------

                    #eyeOpenness = eyeOpennessPoints[eyeOpennessIdx] # static
                    eyeOpenness = 1.0
                    if (simulateBlink):
                        eyeOpenness = randomSampleStratified1D(eyeOpennessPoints[eyeOpennessIdx], eyeOpennessInterval) # stratified randomized

                    # set animated eye defined by gaze and eye openness
                    animateEye(eye_obj,eye_LookAt,gaze,eyeOpenness)

                    effectiveEyeClosedness = bpy.data.meshes['Head_Mesh'].shape_keys.key_blocks['Look_down'].value
                    effectiveEyeOpenness = 1.0 - effectiveEyeClosedness


                    # ----------------------------
                    # IRIS SIZE
                    # ----------------------------

                    #irisSize = gauss(0.5,0.1) # randomized using gaussian : mean, sigma
                    irisSize = 0.5 # static
                    set_iris_size(scl_mesh, irisSize, True)

                    # ----------------------------
                    # PUPIL SIZE
                    # ----------------------------

                    # animate for different pupil sizes
                    # pupilSize = 0.5 # static
                    #pupilSize = pupilAdjustmentPoints[pupilAdjustmentIdx] # dynamic without randomization
                    pupilSize = randomSampleStratified1D(pupilAdjustmentPoints[pupilAdjustmentIdx], pupilAdjInterval) # stratified randomization
                    set_pupil_size(scl_mesh, pupilSize, True)        


                    # ----------------------------
                    # LOG EYE STATE INFO FILE
                    # ----------------------------
                    
                    originalEyeLocation = headLocation + headRot * uprightHeadToEye
                    shiftedEyeLocation = head_obj.location + headRot * uprightHeadToEye
                    log_msg.append('''
                        %s,
                        %.8f, %.8f,
                        %.8f, %.8f, %.8f,
                        %.8f, %.8f, %.8f,
                        %.8f, %.8f, %.8f\n'''
                        %(
                        "L",
                        gaze[0], -gaze[1],
                        originalEyeLocation[0], originalEyeLocation[1], originalEyeLocation[2],
                        shiftedEyeLocation[0], shiftedEyeLocation[1], shiftedEyeLocation[2],
                        pupilSize, irisSize, effectiveEyeOpenness
                        )
                    )
                    

                    # advance frames
                    frameNumber = frameNumber + 1
                

# write gaze state for every frame into description file

# update scene
C.scene.update()

for buf in log_msg: # dump it to csv file at once because csv sometimes gets corrupted if we do it line by line.
    logfile.write(buf)
logfile.close()
    
    

print("\n-----------------script complete-----------------")
