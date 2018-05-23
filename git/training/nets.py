"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from __future__ import print_function
import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image
import os, platform
import pdb
from copy import deepcopy

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def xavier_weights_init(m):
    classname = m.__class__.__name__
    gain = nn.init.calculate_gain('leaky_relu')
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight, gain)
        #nn.init.constant(m.bias, 0.5)
         
def kaiming_weights_init(m):
    classname = m.__class__.__name__
    gain = nn.init.calculate_gain('leaky_relu')
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform(m.weight, gain)
        #nn.init.constant(m.bias, 0.5)

def default_weights_init(m):
    pass

def identity_weights_init(m):
    classname = m.__class__.__name__


class customDropout2D(nn.Dropout2d):
    def __init__(self):
        super

    def forward():
        a = 1 # just to prevent the network to crash
        #custom forward here

#Note from Pytorch docs:
#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#Bias is true by default, so account for that when loading weights into an external program
#torch.save(model.state_dict()) serializes weights and bias to a single file.
class LeanNet(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,16,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(24,36,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(36,54,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(54,81,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(81,122,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2)
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes) # 2440 inputs (122 layers, resolution of 5 x 4)
        self.pad = nn.ReplicationPad2d((1,1,1,1))

    def forward(self,x):
        #logging.debug('Input size: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class ResNet50FeaturesFrozen(torchvision.models.resnet.ResNet):
    def __init__(self, num_output_nodes = 2, input_resolution = (224,224)):
        assert(input_resolution == (224,224))
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        if platform.system() == 'Windows':
            weight_path = 'X:/pretrained_networks/ResNet50.pth'
        elif platform.system() == 'Linux':
            weight_path = '/playpen/data/pretrained_networks/resnet50.pth'
        self.load_state_dict(torch.load(weight_path))
        # freeze existing feature layers
        for param in self.parameters():
            param.requires_grad = False
        # re-configure the last layer. By default newly configured layers have requires_grad = True
        num_filters = self.fc.in_features
        self.fc = nn.Linear(num_filters, num_output_nodes)

    # def forward(self, x):
    #     x = x.repeat(1,3,1,1).clone()
    #     x = super().forward(x)
    #     return x

    def genFeatureMaps(self, x, featureSpacing):
        # Not supported yet - we are working on a pre-trained network
        return None, None

class ResNet50(torchvision.models.resnet.ResNet):
    def __init__(self, num_output_nodes = 2, input_resolution = (224,224)):
        assert(input_resolution == (224,224))
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        if platform.system() == 'Windows':
            weight_path = 'X:/pretrained_networks/ResNet50.pth'
        elif platform.system() == 'Linux':
            weight_path = '/playpen/data/pretrained_networks/resnet50.pth'
        self.load_state_dict(torch.load(weight_path))
        # re-configure the last layer. By default newly configured layers have requires_grad = True
        num_filters = self.fc.in_features
        self.fc = nn.Linear(num_filters, num_output_nodes)

    # def forward(self, x):
    #     x = x.repeat(1,3,1,1).clone()
    #     x = super().forward(x)
    #     return x

    def genFeatureMaps(self, x, featureSpacing):
        # Not supported yet - we are working on a pre-trained network
        return None, None

# A lean net whose number of layer, stride, and number of features are configurable.
class ConfigurableLeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240), strides = (2,2,2,2,2,2), output_channel_counts = (16, 24, 36, 54, 81, 122)):
        super().__init__() #marker
        assert(len(strides) == len(output_channel_counts))
        input_channel_counts = [output_channel_count for output_channel_count in output_channel_counts[0:-1]]
        input_channel_counts.insert(0,1)
        input_channel_counts = tuple(input_channel_counts)
        self.conv = list()
        self.drop = list()
        layer_output_dimensions = list()
        for stride, input_channel_count, output_channel_count in zip(strides, input_channel_counts, output_channel_counts):
            self.conv.append(nn.Conv2d(input_channel_count, output_channel_count, 3, stride = stride))
            self.drop.append(nn.Dropout2d(p = dropout))
            if len(layer_output_dimensions) == 0:
                input_dimension = np.asarray((input_resolution[1], input_resolution[0]))
            else:
                input_dimension = layer_output_dimensions[-1]
            if stride == 1:
                output_dimension = input_dimension - 2
            elif stride == 2:
                output_dimension = np.ceil(input_dimension / 2) - 1
            layer_output_dimensions.append(output_dimension)
        layer_output_dimensions = np.asarray(layer_output_dimensions)
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv[-1].out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes)

    def forward(self,x):
        for conv, drop in zip(self.conv, self.drop):
            x = F.relu(conv(x))
            x = drop(x)
        x = x.view(-1, self.last_layer_node_count)
        x = self.fc1(x)
        return x

    def cuda(self):
        super().cuda()
        for conv, drop in zip(self.conv, self.drop):
            conv.cuda()
            drop.cuda()
        
    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = list()
        featureDimensions = list()
        for conv in self.conv:
            x = F.relu(conv(x))
            featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
            featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class LeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,16,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(24,36,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(36,54,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(54,81,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(81,122,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count,num_output_nodes) # 976 inputs (122 layers, resolution of 4 x 2)

    def forward(self,x):
        #logging.debug('Input size: %s'%(x.size(),))
        x = F.relu(self.conv1(x))
        #pdb.set_trace()
        x = self.drop1(x)
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        #pdb.set_trace()
        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x
        
    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = F.relu(self.conv1(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class DoubleLeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,32,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(32,48,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(48,72,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(72,108,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(108,162,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(162,244,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes) # 2440 inputs (122 layers, resolution of 5 x 4)

    def forward(self,x):
        #logging.debug('Input size: %s'%(x.size(),))
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = F.relu(self.conv1(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class HalfLeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,8,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(8,12,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(12,18,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(18,27,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(27,41,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(41,61,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes) # 2440 inputs (122 layers, resolution of 5 x 4)

    def forward(self,x):
        #logging.debug('Input size: %s'%(x.size(),))
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = F.relu(self.conv1(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class LeanNetZeroPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,16,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(24,36,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(36,54,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(54,81,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(81,122,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2)
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes) # 2440 inputs (122 layers, resolution of 5 x 4)
        self.pad = nn.ZeroPad2d((1,1,1,1))

    def forward(self,x):
        #logging.debug('Input size: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class LeanPoolNet(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,16,3,stride=1)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=1)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(24,36,3,stride=1)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(36,54,3,stride=1)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(54,81,3,stride=1)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(81,122,3,stride=1)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2 - 1)
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        # logging.debug(layer_output_dimensions)
        self.fc1 = nn.Linear(self.last_layer_node_count, num_output_nodes) # 2440 inputs (122 layers, resolution of 5 x 4)

    def forward(self,x):
        # logging.debug('Input size: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop1(x)
        # logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop2(x)
        # logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv3(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop3(x)
        # logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv4(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop4(x)
        # logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv5(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop5(x)
        # logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv6(x)), kernel_size = 2, ceil_mode = True)
        x = self.drop6(x)
        # logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv3(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv4(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv5(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.avg_pool2d(F.relu(self.conv6(x)), kernel_size = 2, ceil_mode = True)
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class DLean2Net(nn.Module):
    '''A lean network'''
    def __init__(self, dropout=0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__()
        self.conv1a = nn.Conv2d(1,16,3,stride=1)
        self.drop1a = nn.Dropout2d(p=dropout)
        self.conv1b = nn.Conv2d(16,16,3,stride=2)
        self.drop1b = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(24,36,3,stride=2)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(36,54,3,stride=2)
        self.drop4 = nn.Dropout2d(p=dropout)
        self.conv5 = nn.Conv2d(54,81,3,stride=2)
        self.drop5 = nn.Dropout2d(p=dropout)
        self.conv6 = nn.Conv2d(81,122,3,stride=2)
        self.drop6 = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        # logging.debug(layer_output_dimensions)
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6.out_channels)
        self.fc1 = nn.Linear(self.last_layer_node_count,num_output_nodes)
        self.pad = nn.ZeroPad2d((1,1,1,1))

    def forward(self,x):
        x = F.relu(self.conv1a(self.pad(x)))
        x = self.drop1a(x)
        # logging.debug('Conv1a output: %s'%(x.size(),))
        x = F.relu(self.conv1b(x))
        x = self.drop1b(x)
        # logging.debug('Conv1b output: %s'%(x.size(),))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        # logging.debug('Conv2 output: %s'%(x.size(),))
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        # logging.debug('Conv3 output: %s'%(x.size(),))
        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        # logging.debug('Conv4 output: %s'%(x.size(),))
        x = F.relu(self.conv5(x))
        x = self.drop5(x)
        # logging.debug('Conv5 output: %s'%(x.size(),))
        x = F.relu(self.conv6(x))
        x = self.drop6(x)
        # logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x

    def genFeatureMaps(self,x,featureSpacing):
        #logging.debug('Input size: %s'%(x.size(),))
        featureMaps = []
        featureDimensions = []
        x = self.pad(x)
        x = F.relu(self.conv1a(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1a output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv1b(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv1b output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 5, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv2 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv3(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 6, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv3 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv4(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 8, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv4 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv5(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 9, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv5 output: %s'%(x.size(),))
        x = self.pad(x)
        x = F.relu(self.conv6(x))
        featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 11, padding = featureSpacing).cpu().numpy()[0])
        featureDimensions.append((x.data.size()[2],x.data.size()[3]))
        #logging.debug('Conv6 output: %s'%(x.size(),))
        # x = x.view(-1, 2440)
        # #logging.debug('FC1 input: %s'%(x.size(),))
        # x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return featureMaps, featureDimensions

class CalibAffine2(nn.Module):
    '''A calibration network of 2x2 affine transform for multi-user calibration'''
    def __init__(self, subjects):
        super().__init__()
        self.user = nn.ModuleList()
        for i in range(subjects):
            fc = nn.Linear(2,2)
            torch.nn.init.eye(fc.weight)
            torch.nn.init.constant(fc.bias,0)
            self.user.append(fc)
    def forward(self,x,subjectID):
        x = self.user[int(subjectID)](x)
        return x

class CalibAffine8(nn.Module):
    '''A calibration network of 2x2 affine transform for multi-user calibration'''
    def __init__(self, subjects):
        super().__init__()
        self.user = nn.ModuleList()
        for i in range(subjects):
            fc = nn.Linear(8,2)
            torch.nn.init.eye(fc.weight)
            torch.nn.init.constant(fc.bias,0)
            self.user.append(fc)
    def forward(self,x,subjectID):
        x = self.user[int(subjectID)](x)
        return x

class CalibNet(nn.Module):
    def __init__(self, network, calibration):
        super().__init__()
        self.add_module('net',network)
        self.add_module('calib',calibration)
    def forward(self,x,subjectID):
        x = self.net(x)
        subjects = np.unique(subjectID)
        output = Variable(torch.zeros(x.size(0),2)).cuda()
        for subject in subjects:
            #print(subject)
            mask = Variable(torch.nonzero(subjectID == int(subject))[:,0]).cuda()
            subset = torch.index_select(x,0,mask)
            outset = self.calib(subset,subject)
            output[mask] = outset
        #print(output.size())
        return output


class BinocularLeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        self.conv1 = nn.Conv2d(1,16,3,stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(16,24,3,stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)

        self.conv3Left = nn.Conv2d(24,36,3,stride=2)
        self.drop3Left = nn.Dropout2d(p=dropout)
        self.conv3Right = nn.Conv2d(24,36,3,stride=2)
        self.drop3Right = nn.Dropout2d(p=dropout)

        self.conv4Left = nn.Conv2d(36,54,3,stride=2)
        self.drop4Left = nn.Dropout2d(p=dropout)
        self.conv4Right = nn.Conv2d(36,54,3,stride=2)
        self.drop4Right = nn.Dropout2d(p=dropout)

        self.conv5Left = nn.Conv2d(54,81,3,stride=2)
        self.drop5Left = nn.Dropout2d(p=dropout)
        self.conv5Right = nn.Conv2d(54,81,3,stride=2)
        self.drop5Right = nn.Dropout2d(p=dropout)


        self.conv6Left = nn.Conv2d(81,122,3,stride=2)
        self.drop6Left = nn.Dropout2d(p=dropout)
        self.conv6Right = nn.Conv2d(81,122,3,stride=2)
        self.drop6Right = nn.Dropout2d(p=dropout)
        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        # Simply multiplying by two is correct here because the convolution path is fully duplicated even though some kernel weights are shared
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6Left.out_channels) * 2
        self.fc1 = nn.Linear(self.last_layer_node_count,num_output_nodes) # 976 inputs (122 layers, resolution of 4 x 2)

    def forward(self,x):
        #Split into left and right eye images.
        left = x[:,0:1,:,:]
        right = x[:,1:2,:,:]

        #Conv1
        left = F.relu(self.conv1(left))
        left = self.drop1(left)

        right = F.relu(self.conv1(right))
        right = self.drop1(right)


        #Conv2
        left = F.relu(self.conv2(left))
        left = self.drop2(left)

        right = F.relu(self.conv2(right))
        right = self.drop2(right)
        #logging.debug('Conv2 output: %s'%(x.size(),))

        #Switch here to separate convolution paths
        #Conv3
        left = F.relu(self.conv3Left(left))
        left = self.drop3Left(left)

        right = F.relu(self.conv3Right(right))
        right = self.drop3Right(right)
        #logging.debug('Conv3 output: %s'%(x.size(),))

        #Conv4
        left = F.relu(self.conv4Left(left))
        left = self.drop4Left(left)

        right = F.relu(self.conv4Right(right))
        right = self.drop4Right(right)
        #logging.debug('Conv4 output: %s'%(x.size(),))

        #Conv5
        left = F.relu(self.conv5Left(left))
        left = self.drop5Left(left)

        right = F.relu(self.conv5Right(right))
        right = self.drop5Right(right)
        #logging.debug('Conv5 output: %s'%(x.size(),))

        #Conv6
        left = F.relu(self.conv6Left(left))
        left = self.drop6Left(left)

        right = F.relu(self.conv6Right(right))
        right = self.drop6Right(right)

        x = torch.cat((left, right), 1)

        #logging.debug('Conv6 output: %s'%(x.size(),))
        x = x.view(-1, self.last_layer_node_count)
        #logging.debug('FC1 input: %s'%(x.size(),))
        x = self.fc1(x)
        #logging.debug('FC1 output: %s'%(x.size(),))
        return x