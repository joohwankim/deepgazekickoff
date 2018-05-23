################################################################################
# This script generates all the necessary scripts for rendering a large-scale
# blender scene with just one click.
# Joohwan Kim (sckim@nvidia.com) Feb 6th 2018
################################################################################

import csv, os, sys, pdb, platform
import numpy as np

#############################################################################################################
# Parameters setup. This should be all you need to change.
#############################################################################################################

num_node = 3 # Across how many SaturnV nodes the rendering of each blend file will be distributed.
APIKey = '64qHFHBLal19kOWcuCbVJ8x3HigGxfeMKswfgudX' # copy and paste your own API key here.
# Joohwan's APIKey: 64qHFHBLal19kOWcuCbVJ8x3HigGxfeMKswfgudX
# Michael's APIKey: e5ahhEwjH8wpcG0oISCZVDtDIf2AA0I53hY5kwfS
# Zander's  APIKey: DCssxyKuHlz52afSLdsviPJQM4zxPw1Vt02Pn3og
# models = [
#     'nxp_male_01','nxp_male_02','nxp_male_03','nxp_male_04','nxp_male_05',
# ]
# models = [
#     'nxp_female_01','nxp_female_02','nxp_female_03','nxp_female_04','nxp_female_05',
#     'nxp_male_01','nxp_male_02','nxp_male_03','nxp_male_04','nxp_male_05',
# ]
models = [
    'nxp_female_04','nxp_female_05',
    # 'nxp_male_03','nxp_male_04','nxp_male_05','nxp_female_02',
]
# plans = [
#     'TestRobustnessToBlink','TestRobustnessToDepthSlippage','TestRobustnessToPupilSize','TestRobustnessToSlippage','TestRobustnessToTransversalSlippage',
# ]
# plans = [
#     'TestRobustnessToPupilSize',
# ]
plans = [
    # 'Plan002',
    'Plan002_no_glint',
]
iter_models = list()
iter_plans = list()
iter_blend_filepaths = list()
for model in models:
    for plan in plans:
        iter_models.append(model)
        iter_plans.append(plan)
        iter_blend_filepaths.append(
            os.path.join('X:/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scenes/'+model+'/scenes/'+model+'_'+plan+'.blend')
        )

#############################################################################################################
# Submit render jobs.
#############################################################################################################

for model, plan, blend_filepath in zip(iter_models, iter_plans, iter_blend_filepaths):
    csv_filepath = blend_filepath.replace('.blend','.csv')
    set_description = os.path.basename(blend_filepath).replace('.blend','')

    # Find out more parameters necessary for generating the scripts.
    with open(csv_filepath,'rt') as f:
        csv_f = csv.reader(f)
        num_images = sum(1 for row in csv_f) - 1 # count number of images
        divide_jobs = np.round(np.linspace(0,num_images,num_node + 1))
        image_count = [divide_jobs[ii+1] - divide_jobs[ii] for ii in range(num_node)]
        image_offset = [divide_jobs[ii] for ii in range(num_node)]

    # create a folder with a dscriptive name. Overwrite if exists already.
    if not os.path.exists(blend_filepath.replace('.blend','')):
        os.mkdir(blend_filepath.replace('.blend',''))
    # except:
    #     print('Folder already exists... Check what you already have.')
    #     sys.exit()

    # generate a shell script that calls blender for executing the rendering.
    render_f = open(os.path.join(blend_filepath.replace('.blend',''), 'render.sh'),'w')
    message = '''
    counter=0
    numInstances=$1
    numFramesPerInstance=$2
    offset=$3
    for counter in `seq 1 $numInstances`;
    do
        startCounter=$((((counter-1) * numFramesPerInstance) + offset))
        endCounter=$((startCounter + (numFramesPerInstance-1)))
        /usr/local/blender/blender /media/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scenes/%s/scenes/%s.blend -b --python /media/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scripts/render_with_cuda.py -s $startCounter -e $endCounter -a > /media/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scenes/%s/scenes/%s/log_$((counter))_$((offset)).txt &

    done
    wait
    echo "script finished"
    '''%(model, set_description, model, set_description)
    # -o /media/scratch/sckim/%s/frame 
    
    # message.replace(os.linesep,'\n')
    # message = message.replace('\r','')
    render_f.write(message)
    render_f.close()
    if platform.system() == 'Windows':
        os.system('dos2unix %s'%(os.path.join(blend_filepath.replace('.blend',''), 'render.sh')))
    else:
        #os.system('''cat %s/render.sh | tr -d '\\r' >> %s/render.sh'''%(set_description,set_description))
        os.system('chmod 777 %s'%(os.path.join(blend_filepath.replace('.blend',''), 'render.sh')))

    # Generate json files, each of which submits a SaturnV cluster job for the rendering.
    for ii in range(num_node):
        json_fn = 'job_' + str(ii+1) + '.json'
        json_f = open(os.path.join(blend_filepath.replace('.blend',''), json_fn),'w')
        message = '''
        {
            "jobDefinition": {
                "name": "%s_%d_%d",
                "description": "",
                "clusterId": 425,
                "dockerImage": "nvidian_research1/blender-gpu5:2.7.9",
                "jobType": "BATCH",
                "command": "/media/scratch/sckim/deep-gaze/code/dataset/NXPeye_with_mounted_glasses/scenes/%s/scenes/%s/render.sh 1 %d %d",
                "resources": {
                    "gpus": 4,
                    "systemMemory": 128,
                    "cpuCores": 2
                },
                "jobDataLocations": [
                    {
                        "mountPoint": "/media",
                        "protocol": "NFSV3",
                        "sharePath": "/export/deep-gaze2.cosmos393",
                        "shareHost": "dcg-zfs-01.nvidia.com"
                    }
                ],
                "portMappings": []
            }
        }'''%(set_description,image_offset[ii],image_offset[ii] + image_count[ii],model,set_description,image_count[ii],image_offset[ii])
        json_f.write(message)
        json_f.close()

    # Generate a shell script that executes all the above json files.
    submitter_fn = 'submit_jobs.sh'
    submitter_f = open(os.path.join(blend_filepath.replace('.blend',''), submitter_fn),'w')
    submitter_f.write('dgx config set -s compute.nvidia.com -k %s\n'%(APIKey))
    for ii in range(num_node):
        # json_fn = os.path.join(set_description,'job_%d.json'%(ii+1))
        json_fn = blend_filepath.replace('.blend','') + '/job_%d.json'%(ii+1)
        submitter_f.write('dgx job submit -f %s\n'%(json_fn))
    submitter_f.close()
    if platform.system() == 'Linux':
        os.system('chmod 777 %s'%(os.path.join(blend_filepath.replace('.blend',''),submitter_fn)))

    # Now execute the submitter shell script
    os.system(os.path.join(blend_filepath.replace('.blend',''),submitter_fn))
