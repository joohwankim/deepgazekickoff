import bpy
from mathutils import Vector, Euler, Matrix

###################################
# GEOMETRICAL CONFIGURATION
###################################
def load_geometrical_configuration(model_name):
    centralEyeLocation = Vector((-0.625579, 1.21064, 3.07758))  # The eye position assumed by the HMD. In reality this does not match with the eye position due to the slippage.
    eyeLookAtOrigin = centralEyeLocation + Vector((0.31, 0.0, 0.0)) # position of the cyclopian eye, assuming IPD of 6.2cm (average)
    # nxp models
    if model_name == 'nxp_female_01':
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

