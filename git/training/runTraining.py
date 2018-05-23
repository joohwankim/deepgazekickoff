"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import argparse, logging, os, dlcore.train, sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job', required=True, help='Which network to train. Specify a folder containing configuration file')
    parser.add_argument('-v', '--var', nargs='*', action='append', help='A varaible and value pair')
    parser.add_argument('-r', '--resume', default=None, help='Address to a checkpoint file. If given, resume training from the checkpoint file.')
    args = parser.parse_args()
    #config = dlcore.train.loadModule(os.path.join(args.job,'config.py'))
    config = dlcore.train.loadModule(args.job)
    if args.var:
        for var in args.var:
            dtype = type(getattr(config, var[0]))
            if len(var) == 2:
                setattr(config, var[0], dtype(var[1]))
    if os.path.abspath(config.result_dir) == os.path.abspath('./'):
        config.result_dir = os.path.normpath(args.job)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #logging.basicConfig(filename=os.path.join(config.result_dir,config.log), level=config.log_level)
    dlcore.train.main(config, args.resume)
