import bpy
import argparse


# set CUDA as render device
bpy.context.user_preferences.addons.get('cycles').preferences.compute_device_type = 'CUDA'

# set cycles as render engine
bpy.data.scenes["Scene"].render.engine = 'CYCLES'

# render using 'GPU compute'
bpy.data.scenes["Scene"].cycles.device = 'GPU'

# set animation area
#bpy.data.scenes["Scene"].frame_start = 
#bpy.data.scenes["Scene"].frame_end = 