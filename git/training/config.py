"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

'''
NOTE: all modules imported must have '_' underscore prefix
'''

#----------------------------------------------------------------------------
# Base directories.
#----------------------------------------------------------------------------

# Hostname and user.
# - Used for reporting, as well as specializing paths on a per-host basis.

import os as _os
import platform as _platform # renaming of the model is necessary for the serializer ignore the modules (which can't be pickled).

# Base directory for input data.

if _platform.system() == 'Linux':
    data_dir = (
        _os.environ['GAZETRACK_DATA_DIR'] if 'GAZETRACK_DATA_DIR' in _os.environ else
        # '/playpen/data/data_desktop/gaze_20170311')
        '/playpen/data/scratch/sckim/blender_output')
        # '/playpen/data/data_real_glasses')
    eval_dir = (
        _os.environ['GAZETRACK_DATA_DIR'] if 'GAZETRACK_DATA_DIR' in _os.environ else
        # '/playpen/data/data_desktop/gaze_20170311')
        # '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v2_VerticalFlipped')
        '/playpen/data/data_real_glasses')
    tensorboard_log_dir = '/playpen/data/scratch/sckim/deep-gaze/tensorboard_logs'
elif _platform.system() == 'Windows':
    data_dir = (
        _os.environ['GAZETRACK_DATA_DIR'] if 'GAZETRACK_DATA_DIR' in _os.environ else
        # 'X:/data_desktop/gaze_20170311')
        'X:/data_real_glasses')
    eval_dir = (
        _os.environ['GAZETRACK_DATA_DIR'] if 'GAZETRACK_DATA_DIR' in _os.environ else
        # '/playpen/data/data_desktop/gaze_20170311')
        # 'X:/data_GearVR_Topfoison/onaxis_gaze_v2_VerticalFlipped')
        'X:/data_real_glasses')
    tensorboard_log_dir = 'X:/scratch/sckim/deep-gaze/tensorboard_logs'

# Directory for storing the results of individual training runs.

dataset_desc        = 'debug_binocular_net_female_01_30e'

result_dir = (
    _os.environ['GAZETRACK_RESULT_DIR'] if 'GAZETRACK_RESULT_DIR' in _os.environ else
    './')

#----------------------------------------------------------------------------
# Dataset configuration.
#----------------------------------------------------------------------------


dataset = {
    'name': 'gaze_set',
    'train': [
        # { 'subject': 'Dummy1', 'file': 'JK2_train_qvga.h5'}
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_Plan001_train_qvga.h5'} # moderately small data
        { 'subject': 'Dummy1', 'file': 'nxp_female_01_Plan002_train_binocular_255x191.h5'},
        # { 'subject': 'Dummy1', 'file': 'michael_1_zip_train_binocular_255x191.h5'}, 
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_short_all_qvga.h5'}, # extremely small data
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_short_all_qvga.h5'}, # extremely small data
    ],
    'test': [
        # { 'subject': 'Dummy1', 'file': 'JK2_test_qvga.h5'}
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_Plan001_test_qvga.h5'} # moderately small data
        { 'subject': 'Dummy1', 'file': 'nxp_female_01_Plan002_test_binocular_255x191.h5'},
        # { 'subject': 'Dummy1', 'file': 'michael_1_zip_test_binocular_255x191.h5'},
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_short_all_qvga.h5'}, # extremely small data
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_short_all_qvga.h5'}, # extremely small data
    ],
    'eval': [
        # { 'subject': 'Dummy1', 'file': 'nxp_female_01_short_all_binocular_255x191.h5'}, # extremely small data
        { 'subject': 'MS', 'file': 'michael_1_zip_all_binocular_255x191.h5'},
        # { 'subject': 'AM', 'file': 'AM_qvga.h5'}, 
        # { 'subject': 'CF', 'file': 'CF_qvga.h5'}, 
        # { 'subject': 'EL', 'file': 'EL_qvga.h5'}, 
        # { 'subject': 'JK', 'file': 'JK_qvga.h5'}, 
        # { 'subject': 'KA', 'file': 'KA_qvga.h5'}, 
        # { 'subject': 'MM', 'file': 'MM_qvga.h5'}, 
        # { 'subject': 'MS', 'file': 'MS_qvga.h5'}, 
        # { 'subject': 'NK', 'file': 'NK_qvga.h5'}, 
        # { 'subject': 'SB', 'file': 'SB_qvga.h5'}, 
        # { 'subject': 'WL', 'file': 'WL_qvga.h5'}, 
    ],
    'labelSlice': ':2'
}

torchPresent = False
try:
    import dlcore.datasets as _datasets
    import dlcore.nets as _nets
    import torch.nn as _nn
    torchPresent = True
except ImportError:
    pass
import logging as _logging
import sys as _sys

run_desc                    = 'gaze'                               # Base name the results directory to be created for current run.
if torchPresent:
    network_type            = _nets.BinocularLeanNetNoPadding                         # a class of network 
    network_init            = _nets.kaiming_weights_init            # method to initalize the weights
    dataset_type            = _datasets.SubjectClipData  
    outputs                 = ['directions']
    # glint_counts, unoccluded_pupil_centers, eye_locations, directions, pupil_occlusion_ratios
    num_output_nodes = 0
    for output in outputs:
        if output == 'directions': num_output_nodes += 2              # Number of output nodes from the network
        elif output == 'glint_counts': num_output_nodes += 1              # Number of output nodes from the network
        elif output == 'unoccluded_pupil_centers': num_output_nodes += 2              # Number of output nodes from the network
        elif output == 'eye_locations': num_output_nodes += 3              # Number of output nodes from the network
        elif output == 'pupil_occlusion_ratios': num_output_nodes += 1              # Number of output nodes from the network
        elif output == 'iris_occlusion_ratios': num_output_nodes += 1              # Number of output nodes from the network
        elif output == 'unoccluded_pupil_bounding_boxes': num_output_nodes += 4              # Number of output nodes from the network
        elif output == 'unoccluded_iris_bounding_boxes': num_output_nodes += 4              # Number of output nodes from the network
        elif output == 'unoccluded_pupil_ellipse_fits': num_output_nodes += 5              # Number of output nodes from the network
        elif output == 'unoccluded_iris_ellipse_fits': num_output_nodes += 5              # Number of output nodes from the network
        elif output == 'unoccluded_pupil_to_iris_ratios': num_output_nodes += 1              # Number of output nodes from the network
        else: print('Specify number of output nodes for the network'), sys.exit()
    input_resolution        = (255, 191)                             # resolution of the input image
    # input_resolution        = (255, 191)                             # resolution of the input image
    # input_resolution        = (320, 240)                             # resolution of the input image

    #train_transform         = _datasets.translateImage              # Function for permuting the images
    #translation_magnitude   = 0                                     # The magnitude of translation if it happens. If not specified, the default is 20 (defined in the function 'translateImage').
    loss_func               = _nn.MSELoss
    #calib_network_type      = _nets.CalibAffine2                    # Disable this and the following lines if training on a single subject.
    #meta_network_type       = _nets.CalibNet                        # Network for training by combining regular and calibration networks
augmentations               = {
                            # key describes augmentation type, value is a vector assigning one value to one region in the mask image
                            # 0: pupil, 1: iris, 2: sclera, 3: skin, 4: glint

                            'raw_intensity_offsets':                [0,  0,   0,   0,   0 ],
                            'raw_intensity_offsets_perturbations':   [10,  40,  30,  30,  0],

                            'contrast_scalings':                    [1,1,1,0.75,1], # region-wise contrast scaling
                            'contrast_scalings_perturbations':       [0.2, 0.2, 0.2, 0.35, 0.2 ], 

                            'blur_radii':                           [
                                                                        1 * input_resolution[0] / 255,
                                                                        1 * input_resolution[0] / 255,
                                                                        2 * input_resolution[0] / 255,
                                                                        3 * input_resolution[0] / 255,
                                                                        1 * input_resolution[0] / 255
                                                                    ],
                            # 'blur_radii_perturbations':            [0,  0,  0,  0,  0],
                            'blur_radii_perturbations':             [
                                                                        1 * input_resolution[0] / 255,
                                                                        1 * input_resolution[0] / 255,
                                                                        2 * input_resolution[0] / 255,
                                                                        3 * input_resolution[0] / 255,
                                                                        1 * input_resolution[0] / 255
                                                                    ],

                            'random_circles':                       [0, 0, 0, 50, 0],
                            'random_circles_perturbations':         [0, 0, 0, 10, 0],


                            'blur_radius_for_region_map':           3, # blur the region map for smooth transition across regions after region-wise augmentation

                            'normalize_input':                      True,
                            'global_gaussian_noise':                2, # there seems to be some gaussian noise in the camera sensor, whose effect is more prominent in pupil region because it is completely dark (pixel value = 0) for synthetic images.
                            'global_contrast_perturbation':         0.2,
                              }
num_epochs                  = 300                                     # Number of epochs to train.
calib_start                 = 99
rampup_length               = 10
rampdown_length             = 30
start_epoch                 = 0                                     # Starting epoch.
batch_size                  = 20                                   # Samples per minibatch.
learning_rate               = 0.0003                                 # Multiplier for updating weights every batch
# weight_decay                = 0.00001                               # weight decay in Adam optimizer
momentum                    = 0.9                                   # Training momentum to prevent getting stuck in local minima
adam_beta1                  = 0.9                                   # Default value.
adam_beta2                  = 0.99                                  # Tends to be more robust than 0.999.
adam_epsilon                = 1e-8                                  # Default value.
dropout                     = 0.2                                  # Proportion dropout
save_file                   = 'network.pth'                         #
checkpoint_file             = 'checkpoint.pth'                      #
checkpoint_frequency        = 10                                    #

log                         = 'train.log'                           #
log_level                   = _logging.INFO                         #
#log_level                   = _logging.DEBUG                        #

cluster_id                  = 425                                   # Compute cluster to use for remote training
docker_image                = 'nvidian_research1/pytorch-h5'        #
max_jobs                    = 20                                    #
jobs_dir                    = '/playpen/data/scratch/jobs'          #
tensorboard_log_desc        = dataset_desc
