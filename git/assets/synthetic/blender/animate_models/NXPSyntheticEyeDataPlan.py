def loadPlan(plan_name):
    plan = None
    if plan_name == 'IrisIntensityTest':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':1,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':1,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.0,0.4), # arbitrary number from -1 to 1. Realistic range is from 0.0 to 0.4 (which should correspond to look down of 0.2 to 1.0)
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(1.0, 1.0), # arbitrary number from -1 to 1. Realistic range is from 0 to 1
                'num_intervals':1,
                'jitter':False,
            },
            'iris_intensity':{
                'range':(0.0, 10.0),
                'num_intervals':100,
                'jitter':False,
            },
            'skin_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
            'sclera_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
        }

    elif plan_name == 'SkinIntensityTest':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':1,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':1,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.0,0.4), # arbitrary number from -1 to 1. Realistic range is from 0.0 to 0.4 (which should correspond to look down of 0.2 to 1.0)
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(1.0, 1.0), # arbitrary number from -1 to 1. Realistic range is from 0 to 1
                'num_intervals':1,
                'jitter':False,
            },
            'iris_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
            'skin_intensity':{
                'range':(0.0, 10.0),
                'num_intervals':100,
                'jitter':False,
            },
            'sclera_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
        }

    elif plan_name == 'ScleraIntensityTest':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':1,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':1,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.0,0.4), # arbitrary number from -1 to 1. Realistic range is from 0.0 to 0.4 (which should correspond to look down of 0.2 to 1.0)
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(1.0, 1.0), # arbitrary number from -1 to 1. Realistic range is from 0 to 1
                'num_intervals':1,
                'jitter':False,
            },
            'iris_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
            'skin_intensity':{
                'range':(0.0, 2.0),
                'num_intervals':1,
                'jitter':False,
            },
            'sclera_intensity':{
                'range':(0.0, 10.0),
                'num_intervals':100,
                'jitter':False,
            },
        }
        
    elif plan_name == 'Plan001':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':10,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':5,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':3,
                'jitter':True,
            },
            'led_strength_instruction': 'as_in_prototype',
            'show_glint': True,
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'Plan001_no_glint':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':10,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':5,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':3,
                'jitter':True,
            },
            'led_strength_instruction': 'as_in_prototype',
            'show_glint': False,
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'Plan002':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':15,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':8,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':4,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':3,
                'jitter':True,
            },
            'led_strength_instruction': 'as_in_prototype',
            'show_glint': True,
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'Plan002_no_glint':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':15,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':8,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':4,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':3,
                'jitter':True,
            },
            'led_strength_instruction': 'as_in_prototype',
            'show_glint': False,
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'TestRobustnessToBlink':
        plan = {
#            'jitter_gaze_per_frame':False, # whether to jitter gaze direction for one sample direction (if False) or for every combination (if True). Assumed True if the variable is undefined.
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':15,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.1,0.1), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(0.0,0.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(0.0,0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':20,
                'jitter':False,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'TestRobustnessToPupilSize':
        plan = {
#            'jitter_gaze_per_frame':False, # whether to jitter gaze direction for one sample direction (if False) or for every combination (if True). Assumed True if the variable is undefined.
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':15,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.0,1.0), # arbitrary number from 0 to 1.
                'num_intervals':20,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(0.0,0.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(0.0,0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(0.6, 0.6), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'TestRobustnessToTransversalSlippage':
        plan = {
#            'jitter_gaze_per_frame':False, # whether to jitter gaze direction for one sample direction (if False) or for every combination (if True). Assumed True if the variable is undefined.
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':15,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.2,0.2), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':5,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(0.0, 0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':5,
                'jitter':False,
            },
            'blink':{
                'range':(0.6, 0.6), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'TestRobustnessToDepthSlippage':
        plan = {
#            'jitter_gaze_per_frame':False, # whether to jitter gaze direction for one sample direction (if False) or for every combination (if True). Assumed True if the variable is undefined.
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':20,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':15,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.2,0.2), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(0.0,0.0), # in mm
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(0.0, 5.0),
                'num_intervals':10,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(0.0,0.0),
                'num_intervals':1,
                'jitter':False,
            },
            'blink':{
                'range':(0.6, 0.6), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'TestRobustnessToSlippage':
        plan = {
#            'jitter_gaze_per_frame':False, # whether to jitter gaze direction for one sample direction (if False) or for every combination (if True). Assumed True if the variable is undefined.
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':10,
                'jitter':False,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':10,
                'jitter':False,
            },
            'pupil_size':{
                'range':(0.2,0.2), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':5,
                'jitter':False,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(0.0, 5.0),
                'num_intervals':5,
                'jitter':False,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':5,
                'jitter':False,
            },
            'blink':{
                'range':(0.6, 0.6), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':False,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'SimplePlan':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':2,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':2,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':2,
                'jitter':True,
            },
            'led_strength_instruction': 'as_in_prototype',
            'show_glint': True,
            # No intensity variations in this plan. Use the default values.
        }
    elif plan_name == 'SimplePlan2':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':2,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':2,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':2,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':2,
                'jitter':True,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    elif plan_name == 'Debugging':
        plan = {
            'gaze_x':{
                'range':(-30,30), # in deg
                'num_intervals':1,
                'jitter':True,
            },
            'gaze_y':{
                'range':(-20,20),
                'num_intervals':1,
                'jitter':True,
            },
            'pupil_size':{
                'range':(0.0,0.5), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':True,
            },
            'slippage_x':{ # left and right
                'range':(-5.0,5.0), # in mm
                'num_intervals':1,
                'jitter':True,
            },
            'slippage_y':{ # back and forth !!!NOTE!!!: This is the 'depth' direction by blender convention
                'range':(-0.0, 5.0),
                'num_intervals':1,
                'jitter':True,
            },
            'slippage_z':{ # up and down
                'range':(-5.0,5.0),
                'num_intervals':1,
                'jitter':True,
            },
            'blink':{
                'range':(0.0, 1.0), # arbitrary number from 0 to 1.
                'num_intervals':1,
                'jitter':True,
            },
            # No intensity variations in this plan. Use the default values.
        }
        
    return plan