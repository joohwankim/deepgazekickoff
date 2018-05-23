"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

#----------------------------------------------------------------------------
# submit.py: Submit runs for offline execution in the GPU cluster.
#----------------------------------------------------------------------------

import os, sys, getpass, argparse, logging, shutil, time, itertools, copy, platform, pdb
import numpy as np
        
#----------------------------------------------------------------------------
def create_json(config,name,command):
    # command = "cp -r /workspace /tmp/; cd /tmp/workspace/tensorflow/models/image/mnist; python convolutional.py",
    jsonFile = '''{
"jobDefinition": {
    "name": "%s",
    "description": "",
    "clusterId": %s,
    "dockerImage": "%s",
    "jobType": "BATCH",
    "command": "%s",
    "resources": {
        "gpus": 4,
        "systemMemory": 400,
        "cpuCores": 1
        },
    "jobDataLocations": [{
            "mountPoint": "/playpen/data",
            "protocol": "NFSV3",
            "sharePath": "/export/deep-gaze2.cosmos393",
            "shareHost": "dcg-zfs-01.nvidia.com"
        }],
    "portMappings": []
    }
}'''%(name,config.cluster_id,config.docker_image,command)
    return jsonFile

def submit_to_saturnv(job, config, diff, name, id, total, resume):
    logging.info('Submitting "%s" task to saturn V cluster.   Job %d of %d'%(name,id+1,total))
    #copy code to temp directory
    try:
        user = getpass.getuser()
    except(KeyError):
        user = 'interactive'
    dirName = '%s-%s'%(name,time.strftime("%Y%b%d_%H%M%S"))
    path = os.path.normpath(config.jobs_dir)
    codeDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dlcore')
    tempDir = os.path.join(path,dirName)
    logging.info('Copying code directory %s to %s'%(codeDir,os.path.join(tempDir,'dlcore')))
    shutil.copytree(codeDir, os.path.join(tempDir,'dlcore'))
    configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),job,'config.py')
    logging.info('Copying config from %s to %s'%(configPath,tempDir))
    shutil.copy(configPath, tempDir)
    # update config with unique values
    with open(os.path.join(tempDir,'config.py'), 'a') as f:
        for k, v in diff.items():
            if k == 'tensorboard_log_desc':
                f.write("%s = '%s'\n"%(k,v))
            else:
                f.write('%s = %s\n'%(k,v))
    #generate command
    if resume == None:
        command = 'cd %s; python -m dlcore.train'%tempDir
    else:
        command = 'cd %s; python -m dlcore.train -r %s'%(tempDir, resume)
    #for k, v in diff.items():
    #    command = '%s -v %s %s'%(command,k,v)
    # command = '%s; python -m dlcore.eval'%command # eval.py is now not necessary because we are using tensorboard.
    jsonStr = create_json(config,name,command)
    # make json file
    jsonFile = os.path.join(tempDir,'%s.json'%name)
    logging.debug('%s\n%s'%(jsonFile,jsonStr))
    with open(jsonFile,'w') as f:
        f.write(jsonStr)
    # submit the job
    #if not interactive:
    if platform.system() == 'Windows':
        run_cmd  = 'C:/Python27/python.exe -m dgx job submit -f %s'%jsonFile
    else:
        run_cmd  = 'dgx job submit -f %s'%jsonFile
    if user == 'interactive':
        run_cmd = '/bin/bash -c "source activate python2; %s; source deactivate"'%run_cmd
    logging.info('Executing: %s' % run_cmd)
    os.system(run_cmd)

def configDiff(orig, config):
    # not going to look for removed items - that is not the point
    diff = {}
    for k,v in config.__dict__.items():
        if not hasattr(orig, k) or getattr(orig,k) != v:
            diff[k] = v
    return diff

def copyModule(old):
    new = type(old)(old.__name__, old.__doc__)
    new.__dict__.update(old.__dict__)
    for k,v in new.__dict__.items():
        if isinstance(v, dict):
            new.__dict__[k] = copy.copy(v)
    return new

def loadModule(file, name='config'):
    if sys.version_info[0] == 3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info[0] == 2:
        import imp
        module = imp.load_source(name,file)
    return module

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job', required=True, help='Which network to train. Specify a folder containing config.py')
    parser.add_argument('-v', '--var', nargs='*', action='append', help='A varaible and value pair or variable and range for multiple submits')
    parser.add_argument('-cv', '--cross', action='store_true', help='Do cross validation on the dataset (submit multiple jobs for each cross).')
    parser.add_argument('-n', '--name', default='', help='Name to append to figure titles')
    parser.add_argument('-r', '--resume', default=None, help='Address to a checkpoint file. If given, resume training from the checkpoint file.')
    args = parser.parse_args()
    #args, unknown = parser.parse_known_args()
    config = loadModule(os.path.join(args.job,'config.py'))
    if args.name != '':
        config.run_desc = args.name
    orig = copyModule(config)
    configs = [copyModule(config)]
    names = []
    if args.var:
        name = ''
        for var in args.var:
            dtype = type(getattr(config, var[0]))
            if len(var) == 2:
                for conf in configs:
                    setattr(conf, var[0], dtype(var[1]))
                setattr(config, var[0], dtype(var[1]))
                if var[0] != 'tensorboard_log_desc':
                    if len(name) < 1:
                        name = '%s_%s_%s' % (config.run_desc, var[0], var[1])
                    else:
                        name = '%s_%s_%s' % (name, var[0], var[1])
        names.append(name)

    if len(names) < 1:
        names.append('%s_default'%config.run_desc)
    if len(configs) > config.max_jobs:
        configs = configs[:config.max_jobs]
    for idx, conf in enumerate(configs):
        if 'localdata' in conf.data_dir:
            print('"localdata" still in data path. Please fix before resubmitting.')
            continue
        #print(conf.dataset)
        diff = configDiff(orig,conf)
        #print(names[idx], diff)
        submit_to_saturnv(args.job,conf,diff,names[idx],idx,len(configs),args.resume)
