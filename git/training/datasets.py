"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from itertools import zip_longest as itzl
import logging
import pdb
import torch
from torch.autograd import Variable
from PIL import Image

def resize_region_map(region_map, size):
    if len(region_map.shape) > 2: # color image
        region_map = Image.fromarray(region_map, mode = 'RGB')
    else: # gray
        region_map = Image.fromarray(region_map)
    region_map = region_map.resize(size, Image.NEAREST)
    region_map = np.asarray(region_map, dtype=np.float32)
    return region_map


def resize_img(img, size):
    # Removing the following two lines. Previously we only dealt with gray images, so more than one color channel means we need to convert the given image to a gray image. But now we also have masks generated using all three color channels. The experimenter has to be responsible for keeping color channel to be one for any gray images.
    # if len(img.shape) > 2:
    #     img = img[:,:,0]

    if len(img.shape) > 2: # color image
        img = Image.fromarray(img, mode = 'RGB')
    else: # gray
        img = Image.fromarray(img)
    img = img.resize(size, Image.BICUBIC)
    img = np.clip(img, 0.0, 255.5)
    img = np.asarray(img, dtype=np.uint8)
    return img


class Dataset():
    """docstring for GazeNetDataLoader"""
    def __init__(self, data_paths, outputs = ['directions'], batch_size = 50, input_resolution = (255,191), flip_horizontal = False):
        # flip_horizontal is an experimental feature for debugging the training dynamics of the network.
        self.batch_size = batch_size
        if type(data_paths).__name__ != 'list':
            temp = list()
            temp.append(data_paths)
            data_paths = temp

        # Initialize clip_dir and 
        for d_idx, data_path in enumerate(data_paths):
            logging.debug('   Reading %d out of %d data sets...'%(d_idx + 1, len(data_paths)))
            f = h5py.File(data_path,'r')

            # copy all the information
            if hasattr(self, 'clip_dir'): 
                self.clip_dir = np.concatenate((self.clip_dir, np.asarray(f['clip_dir'][...])), axis = 0)
                self.clip_range = np.concatenate((self.clip_range, np.asarray(f['clip_range'][...])), axis = 0)
            else:
                self.clip_dir = np.asarray(f['clip_dir'][...])
                self.clip_range = np.asarray(f['clip_range'][...])
            # Array of numpy images for all clips being loaded. Add a channel dimension.
            images = np.asarray(f['images'][...])
            resolution_in_h5 = [images.shape[2], images.shape[1]]

            # do resolution scaling
            if tuple(np.asarray(input_resolution)) != (images.shape[2],images.shape[1]): 
                logging.debug('   Resizing images to the specified resolution...')
                rescaled_images = np.zeros((images.shape[0],input_resolution[1],input_resolution[0]))
                for image_idx in range(images.shape[0]):
                    if image_idx == int(images.shape[0] * 0.25):
                        logging.debug('   25 percent done.')
                    if image_idx == int(images.shape[0] * 0.50):
                        logging.debug('   50 percent done.')
                    if image_idx == int(images.shape[0] * 0.75):
                        logging.debug('   75 percent done.')
                    rescaled_images[image_idx,:,:] = resize_img(np.squeeze(images[image_idx,:,:]), (input_resolution[0], input_resolution[1]))
                logging.debug('   Resizing complete. Copying resized images.')
                if flip_horizontal:
                    rescaled_images = np.flip(rescaled_images, axis = 2)
                images = rescaled_images
            
            # add channel dimension to images
            images = np.expand_dims(images, axis = 1)
            if hasattr(self, 'images'):
                self.images = np.concatenate((self.images, images), axis = 0)
            else:
                self.images = images

            # do resolution scaling for the region maps if they exist
            if 'region_maps' in f:
                region_maps = np.asarray(f['region_maps'][...])
                if tuple(np.asarray(input_resolution)) != (region_maps.shape[2],region_maps.shape[1]): 
                    logging.debug('   Resizing region maps to the specified resolution...')
                    rescaled_region_maps = np.zeros((region_maps.shape[0],input_resolution[1],input_resolution[0]))
                    for map_idx in range(region_maps.shape[0]):
                        if map_idx == int(region_maps.shape[0] * 0.25):
                            logging.debug('   25 percent done.')
                        if map_idx == int(region_maps.shape[0] * 0.50):
                            logging.debug('   50 percent done.')
                        if map_idx == int(region_maps.shape[0] * 0.75):
                            logging.debug('   75 percent done.')
                        rescaled_region_maps[map_idx,:,:] = resize_region_map(np.squeeze(region_maps[map_idx,:,:]), (input_resolution[0], input_resolution[1]))
                    logging.debug('   Resizing complete. Copying resized region_maps.')
                    region_maps = rescaled_region_maps
                # add channel dimension to images
                region_maps = np.expand_dims(region_maps, axis = 1)
                if hasattr(self, 'region_maps'):
                    self.region_maps = np.concatenate((self.region_maps, region_maps), axis = 0)
                else:
                    self.region_maps = region_maps
            else:
                self.region_maps = np.asarray(None)

            try: # dataset with updated structures
                if hasattr(self, 'subjects'):
                    self.subjects = np.concatenate((self.subjects, np.asarray(f['labels']['subjects'])), axis = 0)
                else:
                    self.subjects = np.asarray(f['labels']['subjects'])
                # leave it as a non-np array (dictionary is much easier to read).
                # create labels depending on what 'outputs' is. Possible options are:
                # directions, eye_locations, pupil_sizes, iris_sizes, eye_opennesses, unoccluded_pupil_centers, unoccluded_iris_centers, pupil_occlusion_ratios, glint_counts
                if not hasattr(self, 'all_info'):
                    self.all_info = dict()
                for key in f['labels']:
                    # necessary conversions for resolution change...
                    # NOTE: The lengths of major and minor axes of ellipse fits are not converted to the new resolution (doesn't work for resolution other than the original resolution used for generating the h5 file). These will be inaccurate if resolution change happens.
                    if (key == 'unoccluded_pupil_centers') or (key == 'unoccluded_iris_centers') or (key == 'glint1_centers') or (key == 'glint2_centers') or (key == 'glint3_centers') or (key == 'glint4_centers') or (key == 'glint5_centers'):
                        if key in self.all_info:
                            self.all_info[key] = np.concatenate((self.all_info[key], np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1]])), axis = 0)
                        else:
                            self.all_info[key] = np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1]])
                    elif (key == 'unoccluded_pupil_bounding_boxes') or (key == 'unoccluded_iris_bounding_boxes'):
                        if key in self.all_info:
                            self.all_info[key] = np.concatenate((self.all_info[key], np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1], input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1]])), axis = 0)
                        else:
                            self.all_info[key] = np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1], input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1]])
                    elif (key == 'unoccluded_pupil_ellipse_fits') or (key == 'unoccluded_iris_ellipse_fits'):
                        # NOTE: The major/minor axes and angle of tilt of ellipse fits will be WRONG if rescaling happens.
                        if key in self.all_info:
                            self.all_info[key] = np.concatenate((self.all_info[key], np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1], 1.0, 1.0, 1.0])), axis = 0)
                        else:
                            self.all_info[key] = np.asarray(f['labels'][key]) * np.asarray([input_resolution[0] / resolution_in_h5[0], input_resolution[1] / resolution_in_h5[1], 1.0, 1.0, 1.0])
                    else:
                        if key in self.all_info:
                            if len(np.asarray(f['labels'][key]).shape) == 1:
                                self.all_info[key] = np.concatenate((self.all_info[key], np.expand_dims(np.asarray(f['labels'][key]), axis = 1)), axis = 0)
                            else:
                                self.all_info[key] = np.concatenate((self.all_info[key], np.asarray(f['labels'][key])), axis = 0)
                        else:
                            self.all_info[key] = np.asarray(f['labels'][key])
                    if len(self.all_info[key].shape) == 1:
                        self.all_info[key] = np.expand_dims(self.all_info[key], axis = 1)

                if outputs == None: # by default label is gaze direction
                    if 'directions' in [str(x) for x in f['labels']]: # if this is an h5 dataset with new format
                        self.labels = self.all_info['directions']
                    else: # this is old format
                        self.labels = self.all_info
                else: # when custom label is given.
                    self.labels = list()
                    for o in outputs:
                        if o in self.all_info:
                            if len(self.labels) == 0: # the first labels to be added.
                                exec("self.labels = self.all_info['%s']"%o)
                            else:
                                exec("self.labels = np.concatenate((self.labels, self.all_info['%s']), axis = 1)"%o)
                self.labels = np.asarray(self.labels)
            except: # older format (labels contain only gaze directions)
                if hasattr(self, 'labels'):
                    self.labels = np.concatenate((self.labels, np.asarray(f['labels'][:,0:2])), axis = 0)
                    self.subjects = np.concatenate((self.subjects, np.zeros((self.labels.shape[0],))), axis = 0) # dummy values
                else:
                    self.labels = np.asarray(f['labels'][:,0:2])
                    self.subjects = np.zeros((self.labels.shape[0],)) # dummy values
                self.all_info = None

            
            # If the dataset is a binocular data, merge left and right eye data
            eye_ids = list(self.all_info['eye_ids'].squeeze())
            l_indices = [list_index for list_index, eye_id in enumerate(eye_ids) if eye_id == 0]
            r_indices = [list_index for list_index, eye_id in enumerate(eye_ids) if eye_id == 1]
            binocular_pairing = np.concatenate((np.expand_dims(np.asarray(l_indices), axis = 0), np.expand_dims(np.asarray(r_indices), axis = 0)), axis = 0)

            # define temporary numpy arrays for storing binocularly merged data
            binocular_images = np.zeros((int(self.images.shape[0]/2),2,self.images.shape[2],self.images.shape[3]))
            if len(self.region_maps.shape) == 0:
                binocular_region_maps = np.asarray(None)
            elif len(self.region_maps.shape) == 4:
                binocular_region_maps = np.zeros((int(self.region_maps.shape[0]/2),2,self.region_maps.shape[2],self.region_maps.shape[3]))
            binocular_labels = np.zeros((int(self.labels.shape[0]/2), self.labels.shape[1]))
            binocular_subjects = np.zeros((int(self.subjects.shape[0]/2), ))
            binocular_all_info = dict()
            for key in self.all_info.keys():
                l_key = 'left_' + key
                r_key = 'right_' + key
                array_shape = list(self.all_info[key].shape)
                array_shape[0] = int(array_shape[0]/2) # half because we are splitting info into two (one for left eye and one for right eye)
                binocular_all_info[l_key] = np.zeros(array_shape)
                binocular_all_info[r_key] = np.zeros(array_shape)

            for ii in range(binocular_pairing.shape[1]):
                l_i = binocular_pairing[0,ii]
                r_i = binocular_pairing[1,ii]
                binocular_images[ii,:] = np.concatenate((self.images[l_i], self.images[r_i]), axis = 0)
                if len(self.region_maps.shape) == 4:
                    binocular_region_maps[ii,:] = np.concatenate((self.region_maps[l_i], self.region_maps[r_i]), axis = 0)
                # For labels and subject indices we are copying left eye information. The assumption is that we don't need seperate values for each eye. The code needs to be modified if this assumption does not hold.
                binocular_labels[ii,:] = self.labels[l_i,:]
                binocular_subjects[ii] = self.subjects[l_i]

                # For the variable all_info, we copy everything - the left and right eye info individually.
                for key in self.all_info.keys():
                    l_key = 'left_' + key
                    r_key = 'right_' + key
                    binocular_all_info[l_key][ii,:] = self.all_info[key][l_i,:]
                    binocular_all_info[r_key][ii,:] = self.all_info[key][r_i,:]

            # now stored the merged variables in the member variables
            self.images = binocular_images
            if len(self.region_maps.shape) == 4:
                self.region_maps = binocular_region_maps
            self.labels = binocular_labels
            self.subjects = binocular_subjects
            self.all_info = binocular_all_info

            f.close()

    # Shuffle the images within the input array, discards clip information
    def shuffle(self):
        indices = np.arange(self.images.shape[0])
        np.random.shuffle(indices)
        self.images = self.images[indices,:]
        if len(self.region_maps.shape) > 0:
            self.region_maps = self.region_maps[indices,:]
        if self.all_info is not None:
            for key in self.all_info:
                if np.size(self.all_info[key]) == self.images.shape[0]:
                    self.all_info[key] = self.all_info[key][indices]
                else:
                    self.all_info[key] = self.all_info[key][indices,:]
        if np.size(self.labels) == self.images.shape[0]:
            self.labels = self.labels[indices]
        else:
            self.labels = self.labels[indices,:]

    def __len__(self): # return the number of individual data
        return int(np.ceil(self.images.shape[0] / self.batch_size))

    def __getitem__(self,idx):
        # fetch the data
        i_from = int(self.batch_size * idx)
        i_to = int(np.min([self.images.shape[0],self.batch_size*(idx + 1)]))
        images = Variable(torch.from_numpy(self.images[i_from:i_to].astype(np.float32)), volatile = True).cuda()
        labels = Variable(torch.from_numpy(self.labels[i_from:i_to].astype(np.float32)), volatile = True).cuda()
        subjects = Variable(torch.from_numpy(self.subjects[i_from:i_to]), volatile = True).cuda()
        if len(self.region_maps.shape)>0:
            region_maps = Variable(torch.from_numpy(self.region_maps[i_from:i_to].astype(np.float32)), volatile = True).cuda()
        else:
            region_maps = np.asarray(None) # return empty list if the dataset does not contain region_maps
        return images, region_maps, labels, subjects

class SubjectClipData():
    # legacy class using the pytorch dataloader. Dummy definition for keeping backward compatibility to old config files (old config files may try to define this class. Older version of training code uses the defined class for data loading. However, we will be using our new data loading class).
    def __init__(self):
        self.self_description = 'I am a dummy class'