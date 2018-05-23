"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import logging, sys, shutil, time, argparse, os, importlib.util
from PIL import Image
import platform as _platform

# debugging import
import pdb
import numpy as np
import math
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import time


spec = importlib.util.spec_from_file_location('datasets', os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets.py'))
datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasets)
spec = importlib.util.spec_from_file_location('preprocess', os.path.join(os.path.dirname(os.path.realpath(__file__)),'preprocess.py'))
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
#Tensorboard
spec = importlib.util.spec_from_file_location('TensorBoardLogger', os.path.join(os.path.dirname(os.path.realpath(__file__)),'TensorBoardLogger.py'))
TensorBoardLogger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TensorBoardLogger)


def set_optimal_affine_xform(raw_values, targets):
    assert raw_values.shape == targets.shape and targets.shape[0] >= 3
    P = np.concatenate((raw_values, np.ones((raw_values.shape[0],1))), axis = 1)
    Q = np.concatenate((targets, np.ones((targets.shape[0],1))), axis = 1)
    M = np.dot(np.dot(Q.T, P), np.linalg.inv(np.dot(P.T, P)))
    return np.transpose(M, (1,0)) # multiply this to the raw_values to get the optimally Affine-transformed to match with targets

def adjust_learning_rate(optimizer, config, epoch):
    """Ramp up and ramp down of learning rate"""
    if epoch < config.rampup_length:
        p = max(0.0, float(epoch)) / float(config.rampup_length)
        p = 1.0 - p
        rampup_value = math.exp(-p*p*5.0)
    else:
        rampup_value = 1.0
    if epoch >= (config.num_epochs - config.rampdown_length):
        ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
        rampdown_value = math.exp(-(ep * ep) / config.rampdown_length)
    else:
        rampdown_value = 1.0
    adjusted_learning_rate = rampup_value * rampdown_value * config.learning_rate
    logging.debug('   Learning rate: %s '%adjusted_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adjusted_learning_rate

def save_checkpoint(state, isBest, checkFile, bestFile):
    torch.save(state, checkFile)
    if isBest:
        shutil.copyfile(checkFile, bestFile)

def imagesc(img, filename = 'imagesc.png'): # for debugging
    absmax = np.max([np.abs(np.max(img)),np.abs(np.min(img))])
    plt.imshow(img,cmap = 'seismic', vmin = -absmax, vmax = absmax)
    # plt.colorbar()
    plt.savefig(filename)

def showSrcAndSMap(src,smap,filename = 'imagesc.png'):
    plt.subplot(131)
    plt.imshow(src,cmap = 'gray', vmin = -1, vmax = 1)
    plt.subplot(132)
    absmax = np.max([np.abs(np.max(smap)),np.abs(np.min(smap))])
    plt.imshow(smap,cmap = 'plasma', vmin = 0, vmax = absmax)
    plt.subplot(133)
    plt.imshow(smap,cmap = 'plasma', vmin = 0, vmax = 0.2)
    plt.savefig(filename)

def runTraining(net, train, test, lossFunc, optimizer, config, eval_sets = None, **kwargs):
    if kwargs['resume'] == None: # New training. Start fresh.
        bestLoss = float('inf')
        allLosses = []
        allEpochs = [[],{}]
        first_epoch = 0
    else:
        bestLoss = kwargs['bestLoss']
        allLosses = kwargs['allLosses']
        allEpochs = kwargs['allEpochs']
        first_epoch = kwargs['first_epoch']

    # Initialize the preprocessor.
    P = preprocess.Preprocessor(config)

    startTime = time.time()
    # calibration is not yet handled for resuming training
    calibration = False
    calibLosses = []
    calibEpochsPerEpoch = math.ceil(config.num_epochs/(config.num_epochs-config.calib_start))
    calibEpoch = {}
    last_epoch = first_epoch + config.num_epochs
    epoch_calib_start = first_epoch + config.calib_start
	
    if len(config.tensorboard_log_desc) > 0:
        loggerTF = TensorBoardLogger.TensorBoardLogger(True, "", os.path.join(config.tensorboard_log_dir, config.tensorboard_log_desc))
    else:
        loggerTF = TensorBoardLogger.TensorBoardLogger(True, "", config.tensorboard_log_dir)
	
    
    for epoch in range(first_epoch, last_epoch):
        adjust_learning_rate(optimizer, config, epoch - first_epoch)
        isBest = False
        runningLoss = 0.0
        epochStart = time.time()
        preprocessingTimeCounter = 0
        net.train() # activate dropouts

        smap_drawn = False
        logging.debug('   Shuffling the data...')
        train.shuffle() # shuffle the data
        logging.debug('   Shuffling ended. Starting training loop.')
        for ii in range(len(train)):
            # get the inputs
            inputs, region_maps, labels, subjects = train[ii]

            preprocessingStartTime = time.time()
            inputs = P.preprocess(inputs,region_maps,labels,True)
            preprocessingTimeCounter += time.time() - preprocessingStartTime

            # clone and turn on requires_grad
            inputs = Variable(inputs.data.clone(),requires_grad = True, volatile = False)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            trainLoss = lossFunc(outputs, labels)
            trainLoss.backward()
            # scheduler.step()
            optimizer.step()
            runningLoss += trainLoss.data[0] * outputs.shape[0] * outputs.shape[1] # Multiply by output dimension because pytorch MSELoss divides by total number of elements, not sample size
            allLosses.append(trainLoss.data[0])
        runningLoss = runningLoss/train.images.shape[0]
        allEpochs[0].append(runningLoss)

        # There was a nested for loop for measuring test loss. Main reason for the nested structure was how calibration was handled. I (Joohwan) is re-writing it and for now not worrying about calibration on multiple subjects. Later we should add the calibration feature but should simplify the structure as well.
        net.eval() # disable drop-outs
        testLoss = 0.0
        # reportedLoss = ''
        smap_drawn = False
        for ii in range(len(test)):
            inputs, region_maps, labels, subjects = test[ii]
            inputs = P.preprocess(inputs,region_maps,labels,False)

            # render a sensitivity map for debugging...
            if not smap_drawn:
                temp_input = Variable(inputs[0:1].data.clone().cuda(), requires_grad = True)
                smap = np.zeros(temp_input.data.cpu().numpy().shape)
                for ii in range(config.num_output_nodes):
                    temp_output = net(temp_input)
                    temp_output[0,ii].backward()
                    smap += np.squeeze(temp_input.grad.data.cpu().numpy()) ** 2
                    temp_input.grad.zero_()
                smap = np.sqrt(smap)
                if temp_input.shape[1] == 1:
                    loggerTF.log_image({
                        'input_data':(temp_input.data.cpu().numpy()[0,0] + 1) * 255 / 2,
                        'smap':(smap[0,0] / 0.5)
                    })
                elif temp_input.shape[1] == 2:
                    loggerTF.log_image({
                        'input_data_L':(temp_input.data.cpu().numpy()[0,0] + 1) * 255 / 2,
                        'smap_L':(smap[0,0] / 0.5)
                    })
                    loggerTF.log_image({
                        'input_data_R':(temp_input.data.cpu().numpy()[0,1] + 1) * 255 / 2,
                        'smap_R':(smap[0,1] / 0.5)
                    })
                smap_drawn = True

            outputs = net(inputs)
            testLoss += lossFunc(outputs, labels).data[0] * outputs.shape[0] * outputs.shape[1] # Multiply by output dimension because pytorch MSELoss divides by total number of elements, not sample size
        testLoss = testLoss/test.images.shape[0]
        # reportedLoss = '%s %.4f '%(reportedLoss, testLoss)
        if 'test' not in allEpochs[1]:
            allEpochs[1]['test'] = []
        allEpochs[1]['test'].append(testLoss)

        # run the network over evaluation set.
        # By the way, I am not sure whether the name 'evaluation set' is the best description.
        # May need to change the variable name for a more appropriate word.
        if 'eval' in config.dataset and config.outputs[0] == 'directions':
            evalLoss = np.zeros(len(config.dataset['eval']))
            for e_i, eval_set in enumerate(eval_sets):
                outputs = list()
                labels = list()
                for ii in range(len(eval_set)):
                    inputs, region_maps, t_labels, subjects = eval_set[ii]

                    outputs.extend(net(P.preprocess(inputs,region_maps,labels,False)))
                    labels.extend(list(t_labels.data.cpu().numpy()))

                # convert to numpy
                outputs = np.asarray([output.data.cpu().numpy() for output in outputs])
                labels = np.asarray(labels)
                if config.outputs[0] == 'directions': # apply affine transformation
                    global T
                    T = set_optimal_affine_xform(outputs, labels)
                    outputs_homogeneous_coord = np.concatenate((outputs, np.ones((outputs.shape[0],1))), axis = 1)
                    transformed_outputs_homogeneous_coord = np.matmul(outputs_homogeneous_coord, T)
                    outputs = np.divide(transformed_outputs_homogeneous_coord[:,0:2], np.repeat(transformed_outputs_homogeneous_coord[:,2:3], 2, axis = 1))
                outputs = Variable(torch.FloatTensor(outputs).cuda(), volatile = True)
                labels = Variable(torch.FloatTensor(labels).cuda(), volatile = True)
                evalLoss[e_i] = lossFunc(outputs, labels).data[0] * outputs.data.cpu().numpy().shape[1] # Multiply by output dimension because pytorch MSELoss                 

            if 'eval' not in allEpochs[1]:
                allEpochs[1]['eval'] = []
            allEpochs[1]['eval'].append(evalLoss)

        # print statistics
        if testLoss < bestLoss:
            bestLoss = testLoss
            isBest = True
        # save the current weights - every epoch?
        if (epoch-first_epoch+1) % config.checkpoint_frequency == 0 or isBest or (epoch-first_epoch+1)==config.num_epochs:
            calib = {}
            save_checkpoint({ # calibration not implemented yet.
                'epoch': epoch + 1,
                'all_loss': allLosses,
                'all_epochs': allEpochs,
                'config': {k: config.__dict__[k] for k in [i for i in dir(config) if i[:1] != '_']},
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss': bestLoss,
                'calibration': calib,
            }, isBest, os.path.join(os.path.dirname(config.result_dir), config.dataset_desc + '_' + config.checkpoint_file), os.path.join(os.path.dirname(config.result_dir), config.dataset_desc + '_' + config.save_file))
        trainingLossRMSE = np.sqrt(runningLoss)
        testLossRMSE = np.sqrt(testLoss)
        evalLossMeanRMSE = np.nan
        if 'eval' in config.dataset and config.outputs[0] == 'directions':
            evalLossMeanRMSE = np.mean(np.sqrt(evalLoss))
        logging.info('  Training Epoch %d:   Running Time- %.2f     Preprocessing Time- %.2f    Training Loss (RMSE)- %.4f    Test Loss (RMSE)- %.4f      Eval Loss (RMSE)- %.4f' % (epoch + 1, time.time()-epochStart, preprocessingTimeCounter, trainingLossRMSE, testLossRMSE, evalLossMeanRMSE))

        summary_loss_dict = {"train_loss": trainingLossRMSE, "test_loss": testLossRMSE, "validation_loss": evalLossMeanRMSE}
        individual_loss_dict = dict()
        for idx, individual_loss in enumerate(evalLoss):
            individual_loss_dict['vloss_ind_' + config.dataset['eval'][idx]['subject']] = np.sqrt(individual_loss)

        loggerTF.log_dict(summary_loss_dict)
        loggerTF.log_dict(individual_loss_dict)
        #print('  Training Epoch %d:   Running Time- %.2f    Instant Loss- %.3f    Running Loss- %.3f    Test Loss- %.3f' % (epoch + 1, time.time()-epochStart, trainLoss.data[0], runningLoss/len(train), testLoss))
    logging.info('  Training total running time: %.2f seconds'%(time.time()-startTime))

def main(config, resume=None):
    if hasattr(config,'outputs'):
        outputs = config.outputs
    else:
        outputs = ['directions']

    if hasattr(config, 'flip_horizontal'):
        flip_horizontal = config.flip_horizontal
    else:
        flip_horizontal = False

    logging.info('  Loading training data...')
    # get the data. The output is a list of dataset of the [config.dataset_type] type.
    train_data_paths = [os.path.join(config.data_dir, dataset['file']) for dataset in config.dataset['train']]
    train = datasets.Dataset(train_data_paths, outputs, config.batch_size, input_resolution = config.input_resolution, flip_horizontal = flip_horizontal)
    logging.info('  %d images loaded.'%train.images.shape[0])

    logging.info('  Loading test data...')
    test_data_paths = [os.path.join(config.data_dir, dataset['file']) for dataset in config.dataset['test']]
    test = datasets.Dataset(test_data_paths, outputs, config.batch_size, input_resolution = config.input_resolution, flip_horizontal = flip_horizontal)
    logging.info('  %d images loaded.'%test.images.shape[0])

    if 'eval' in config.dataset and config.outputs[0] == 'directions':
        logging.info('  Loading eval data...')
        eval_sets = list()
        for e_i, e in enumerate(config.dataset['eval']):
            eval_path = os.path.join(config.eval_dir, e['file'])
            eval_sets.append(datasets.Dataset(eval_path, outputs, config.batch_size, input_resolution = config.input_resolution, flip_horizontal = flip_horizontal))
            logging.info('  eval set %d (out of %d sets), %d images loaded.'%(e_i + 1, len(config.dataset['eval']), eval_sets[-1].images.shape[0]))

    logging.info('  Data loading complete.  Initializing network.')

    # load network - initialize values (esp if resuming)
    #Net refers to main training network, not metanetwork
    temp_num_output_nodes = 2
    temp_input_resolution = (255,191)
    if hasattr(config, 'num_output_nodes'):
        temp_num_output_nodes = config.num_output_nodes
    if hasattr(config, 'input_resolution'):
        temp_input_resolution = config.input_resolution
    if config.network_type.__name__ == 'ConfigurableLeanNetNoPadding':
        net = config.network_type(dropout=config.dropout, num_output_nodes = temp_num_output_nodes, input_resolution = temp_input_resolution, strides = config.strides, output_channel_counts = config.output_channel_counts)
    elif config.network_type.__name__ == 'ResNet50':
        net = config.network_type(num_output_nodes = temp_num_output_nodes, input_resolution = temp_input_resolution)
    else:
        net = config.network_type(dropout=config.dropout, num_output_nodes = temp_num_output_nodes, input_resolution = temp_input_resolution)
    logging.info('  Network built.  Initializing weights.')
    net.apply(config.network_init)
    logging.info('  Weights initialized.')

    logging.info('  Transferring to graphics card.')
    net.cuda()
    logging.info('  Network initialized.')
    # setup loss function
    logging.info('  Setting up loss and optimizer.')
    lossFunc = config.loss_func()
    #optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum)

    # def lr_adjuster(epoch):
    #     """Ramp up and ramp down of learning rate"""
    #     if epoch < config.rampup_length:
    #         p = max(0.0, float(epoch)) / float(config.rampup_length)
    #         p = 1.0 - p
    #         rampup_value = math.exp(-p*p*5.0)
    #     else:
    #         rampup_value = 1.0
    #     if epoch >= (config.num_epochs - config.rampdown_length):
    #         ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
    #         rampdown_value = math.exp(-(ep * ep) / config.rampdown_length)
    #     else:
    #         rampdown_value = 1.0
    #     adjusted_learning_rate = rampup_value * rampdown_value * config.learning_rate
    #     logging.debug('   Learning rate: %s '%adjusted_learning_rate)
    #     return adjusted_learning_rate

    if config.network_type.__name__ == 'ResNet50':
        params_to_train = net.fc
    else:
        params_to_train = net
    
    if hasattr(config, 'weight_decay'):
        optimizer = optim.Adam(params_to_train.parameters(), lr=config.learning_rate, betas=[config.adam_beta1,config.adam_beta2], eps=config.adam_epsilon, weight_decay = config.weight_decay)
    else:
        optimizer = optim.Adam(params_to_train.parameters(), lr=config.learning_rate, betas=[config.adam_beta1,config.adam_beta2], eps=config.adam_epsilon)
    # scheduler = optim.LambdaLR(optimizer, lr_adjuster)

    # To be implemented (was implemented in earlier version but removed while revising the data loading parts):
    # 1. Initialization of calibration network - meta network combining a calibration and feature networks
    # 2. Resume feature with fine-tuning schematics

    # train network
    logging.info('  Running training.')
    runTraining(**locals())

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--var', nargs='*', action='append', help='A variable and value pair')
    parser.add_argument('-r', '--resume', default=None, help='Address to a checkpoint file. If given, resume training from the checkpoint file.')
    parser.add_argument('-f', '--fine_tune', default=None, help='Address to a checkpoint file. If given, resume training from the checkpoint file.')
    args = parser.parse_args()
    config = loadModule('config.py')
    if args.var:
        for var in args.var:
            dtype = type(getattr(config, var[0]))
            if len(var) == 2:
                setattr(config, var[0], dtype(var[1]))
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.basicConfig(filename=os.path.join(config.result_dir,config.log), level=config.log_level)
    main(config, args.resume)

