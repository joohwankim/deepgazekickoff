"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os, sys, argparse, platform, csv, time, re
import numpy as np
import scipy.signal, scipy.misc
from scipy.interpolate import interp1d
import PIL.Image
import glob
import h5py
import codecs
import pdb
import time
import random
import math
import cv2
import itertools

#----------------------------------------------------------------------------

glint_bbox_size = 16
glint_bbox_max_motion = 16
top_curtain = 150
half_dataset = False

#----------------------------------------------------------------------------
# Creates h5 files out of zip files
class ZipClip:
    from zipfile import ZipFile

    def __init__(self, dir, path):
        self.dir = dir
        self.path = path
        self.fn = []
        self.cached_images = []
        with self.ZipFile(self.path, 'r') as f:
            self.fn = f.namelist()
            self.fn.sort()       
        if half_dataset:
            self.fn = self.fn[::2]
        self.labels = {
            'directions':list(),
            'subjects':list(),
            'eye_ids':list(),
        }
        for fn in self.fn:
            s = fn.replace('.png', '').split('_')
            self.labels['directions'].append((float(s[s.index('X')+1]), float(s[s.index('Y')+1])))
            self.labels['subjects'].append(0.)
            self.labels['eye_ids'].append(s[1])
            #self.labels.append((float(s[s.index('X')+1]), float(s[s.index('Y')+1]), float(s[s.index('B')+1])))
        #self.labels = [self.dir] * len(self.fn)
        self.cached_images = [np.asarray(None)] * len(self.fn) 

    def get_image_and_mask(self, idx):
        from io import BytesIO
        
        if len(self.cached_images[idx].shape) == 0:
            with self.ZipFile(self.path, 'r') as f:
                buf = f.open(self.fn[idx], 'r').read()
                self.cached_images[idx] = np.asarray(PIL.Image.open(BytesIO(buf)), dtype=np.uint8)
                #img = np.asarray(PIL.Image.open(BytesIO(buf)), dtype=np.uint8)
                #self.cached_images[idx] = (img.astype(np.float32)/255.)*2 - 1   # Normalize uint8 values between 0 and 1
        return self.cached_images[idx].astype(np.float32), np.asarray(None), np.asarray(None) # The last two None's are for maintaining consistency of output format with ImageClip iterator

    def finalize_loading(self):
        pass

    def __len__(self):
        return len(self.fn)

    def __getitem__(self, idx):
        return self.get_image_and_mask(idx)

    def __iter__(self):
        return (self.get_image_and_mask(idx) for idx in range(len(self.fn)))

#----------------------------------------------------------------------------

class ZipDataset:
    def __init__(self, path, which_eye = 'left'):
        self.which_eye = which_eye
        self.path = path # the path where all the zip files are stored.
        self.clips = [] # each element is a ZipClip object

    def init(self):
        files = glob.glob(os.path.join(self.path, '*.zip'))
        files.sort()
        # dir_fn is a list. Each element is a tuple of dir (tuple of horizontal and vertical directions) and file name.
        dir_fn = []
        for fn in files:
            s = os.path.basename(fn).replace('.zip', '').split('_')
            #if (self.which_eye == 'left' and s[1] == 'L') or (self.which_eye == 'right' and s[1] == 'R') or self.which_eye == 'binocular':
            for sPart in s:
                if (sPart == 'L' and self.which_eye == 'left') or (sPart == 'R' and self.which_eye == 'right') or ((sPart == 'L' or sPart == 'R') and self.which_eye == 'binocular'):
                    try:
                        if 'X' in s and 'Y' in s:
                            dir = (float(s[s.index('X') + 1]), float(s[s.index('Y') + 1]))
                        elif s[0] == 'From' and s[3] == 'To':
                            dir = (float(s[1]), float(s[2]), float(s[4]), float(s[5]))
                    except Exception as e:
                        print("Cannot parse filename '%s'." % os.path.basename(fn))
                        print(e)
                        exit()
                    dir_fn.append((dir, fn))
                    break
        dir_fn.sort(key=lambda x:x[1])
        self.clips = [ZipClip(dir, fn) for dir, fn in dir_fn]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx]

    def __iter__(self):
        return (clip for clip in self.clips)

#----------------------------------------------------------------------------

# Dataset composed of image files and a csv file that describes the image files.
# This class has to find out on its own which images belong to the same clip from the information in the CSV file.
# First fetch all the information in the csv file and sort it by gaze direction.
# Then put images into the same clip if they share the same gaze direction.
class ImageClip: 

    def __init__(self, dir, img_filenames, skinlessRegionMask_filenames, regionMask_filenames, eye_locations, pupil_sizes, iris_sizes, eye_opennesses, eye_ids):
        if eye_ids[0] == 'R':
            dir = (- dir[0], dir[1])
            eye_locations = [(- eye_locations[0][0], eye_locations[0][1], eye_locations[0][2])]
        self.dir = dir
        self.img_filenames = img_filenames
        self.regionMask_filenames = regionMask_filenames
        self.skinlessRegionMask_filenames = skinlessRegionMask_filenames
        self.labels = {
            'directions':[(dir[0],dir[1])] * len(img_filenames), # Tuple containing x-direction, y-direction, and the third entry is set to be the same as in ZipClip class (0).
            'subjects':[0.] * len(img_filenames),
            'eye_ids':eye_ids,
            'eye_locations':eye_locations,
            'pupil_sizes':pupil_sizes,
            'iris_sizes':iris_sizes,
            'eye_opennesses':eye_opennesses,
        }
        self.cached_images = [np.asarray(None)] * len(self.img_filenames)
        self.cached_regionMasks = [np.asarray(None)] * len(self.img_filenames) # this will stay None if mask doesn't exist
        self.cached_skinlessRegionMasks = [np.asarray(None)] * len(self.img_filenames) # this will stay None if mask doesn't exist

    def get_image_and_mask(self, idx):
        from io import BytesIO
        if len(self.cached_images[idx].shape) == 0:
            self.cached_images[idx] = np.asarray(PIL.Image.open(self.img_filenames[idx]), dtype=np.float32)
            # Flip horizontally if right eye image drawn by blender.
            if self.labels['eye_ids'][idx] == 'R':
                self.cached_images[idx] = np.flip(self.cached_images[idx],1)
        if len(self.cached_regionMasks[idx].shape) == 0 and len(self.regionMask_filenames) > 0:
            self.cached_regionMasks[idx] = np.asarray(PIL.Image.open(self.regionMask_filenames[idx]), dtype=np.float32)
            # Flip horizontally if right eye image drawn by blender.
            if self.labels['eye_ids'][idx] == 'R':
                self.cached_regionMasks[idx] = np.flip(self.cached_regionMasks[idx],1)
        if len(self.cached_skinlessRegionMasks[idx].shape) == 0 and len(self.skinlessRegionMask_filenames) > 0:
            self.cached_skinlessRegionMasks[idx] = np.asarray(PIL.Image.open(self.skinlessRegionMask_filenames[idx]), dtype=np.float32)
            # Flip horizontally if right eye image drawn by blender.
            if self.labels['eye_ids'][idx] == 'R':
                self.cached_skinlessRegionMasks[idx] = np.flip(self.cached_skinlessRegionMasks[idx],1)
        return self.cached_images[idx], self.cached_regionMasks[idx], self.cached_skinlessRegionMasks[idx]

    def finalize_loading(self):
        pass

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        return self.get_image_and_mask(idx)

    def __iter__(self):
        return (self.get_image_and_mask(idx) for idx in range(len(self.img_filenames)))

#----------------------------------------------------------------------------

# Dataset composed of image files and a csv file that describes the image files.
# This class has to find out on its own which images belong to the same clip from the information in the CSV file.
# First fetch all the information in the csv file and sort it by gaze direction.
# Then put images into the same clip if they share the same gaze direction.
class ImageClipDataset:
    def __init__(self, path, which_eye = 'left'):
        self.which_eye = which_eye
        self.path = path # the path where all the image files are stored.
        self.clips = [] # each element is a SingleImageClip object
        self.hasMasks = False
        if len(glob.glob(os.path.join(path,'*mask*'))) > 0:
            self.hasMasks = True

    def unifyDescription(self,key): # Unify description of parameters
        if key == 'GAZE_X':
            key = 'GAZE_DEGREE_X'
        elif key == 'GAZE_Y':
            key = 'GAZE_DEGREE_Y'
        elif key == 'EYE_OPENNESS' or key == 'BLINK':
            key = 'EYEOPENNESS'
        elif key == 'PUPIL_SIZE':
            key = 'PUPILSIZE'
        return key

    def init(self):
        data_desc_filename = glob.glob(os.path.join(self.path,'*.csv'))[0] # csv file that contains description of data files

        # Create a list that contains description of all the images planned to be rendered. Each entry is a dictionary.
        desc_list = list()
        with open(data_desc_filename) as d_f:
            desc_reader = csv.reader(d_f)
            desc_key_list = [] # list of keys of descriptions
            for i, row in enumerate(desc_reader):
                if row[0] == 'EYE': # The first row.
                    # Parameter description. Unify the description.
                    s = list()
                    desc_key_list = [self.unifyDescription(s) for s in row] # Creating list of all the attributes contained in the description file.
                elif not (row[0] == 'IGNORE'): # 'IGNORE' means data is corrupted... skip in that case.
                    try:
                        desc_one_image = dict((desc_key_list[i].replace(' ',''),s) for i, s in enumerate(row)) # a dictionary that contains all the description values in the csv file.
                        desc_one_image['IMG_FILENAME'] = '' # To be filled later while going through the list of files that actually exist. This makes the procedure more robust (doesn't crash when data generation is incomplete for any reason).
                        if self.hasMasks:
                            desc_one_image['REGION_MASK_FILENAME'] = ''
                            desc_one_image['SKINLESS_REGION_MASK_FILENAME'] = ''
                        desc_list.append(desc_one_image)
                    except:
                        pdb.set_trace()

        # Create a list of file names that contains all the images that exist.
        img_filenames = glob.glob(os.path.join(self.path, '*img*.png'))
        # Fill in the file names
        for img_filename in img_filenames:
            img_index = int(re.sub("[^0-9]", "", os.path.basename(img_filename)))
            # img_index = int(os.path.basename(img_filename).replace('frame','').replace('Image','').replace('.png',''))
            if img_index < len(desc_list):
                desc_list[img_index]['IMG_FILENAME'] = img_filename

        if self.hasMasks:
            # Create a list of file names that contains all the images that exist.
            regionMask_filenames = glob.glob(os.path.join(self.path, '*maskWithSkin*.png'))
            # Fill in the file names
            for regionMask_filename in regionMask_filenames:
                regionMask_index = int(re.sub("[^0-9]", "", os.path.basename(regionMask_filename)))
                if regionMask_index < len(desc_list):
                    desc_list[regionMask_index]['REGION_MASK_FILENAME'] = regionMask_filename
            # Create a list of file names that contains all the images that exist.
            skinlessRegionMask_filenames = glob.glob(os.path.join(self.path, '*maskWithoutSkin*.png'))
            # Fill in the file names
            for skinlessRegionMask_filename in skinlessRegionMask_filenames:
                skinlessRegionMask_index = int(re.sub("[^0-9]", "", os.path.basename(skinlessRegionMask_filename)))
                if skinlessRegionMask_index < len(desc_list):
                    desc_list[skinlessRegionMask_index]['SKINLESS_REGION_MASK_FILENAME'] = skinlessRegionMask_filename

        # Only leave valid entries (image should exist)
        desc_list = [desc for desc in desc_list if not (desc['IMG_FILENAME'] == '')]
        if self.hasMasks: # if this set is supposed to have masks, masks should also exist
            desc_list = [desc for desc in desc_list if not (desc['REGION_MASK_FILENAME'] == '')]
            desc_list = [desc for desc in desc_list if not (desc['SKINLESS_REGION_MASK_FILENAME'] == '')]

        if not (self.which_eye == 'binocular'): # for monocular sets, only leave the eye we are interested.
            desc_list = [desc for desc in desc_list if (desc['EYE'] == 'L' and self.which_eye == 'left') or (desc['EYE'] == 'R' and self.which_eye == 'right')]

        # At this point all the information should be complete (no empty field).

        # Sort it with respect to gaze direction
        desc_list.sort(key = lambda x:(float(x['GAZE_DEGREE_X']),float(x['GAZE_DEGREE_Y'])))

        # For each sequence of images sharing the same gaze direction, add them as a clip.
        temp_img_filenames = list()
        temp_regionMask_filenames = list()
        temp_skinlessRegionMask_filenames = list()
        temp_eye_locations = list()
        temp_pupil_sizes = list()
        temp_iris_sizes = list()
        temp_eye_opennesses = list()
        temp_eye_ids = list()
        temp_dir = (float(desc_list[0]['GAZE_DEGREE_X']),float(desc_list[0]['GAZE_DEGREE_Y']))
        for i, desc in enumerate(desc_list):
            next_dir = (float(desc['GAZE_DEGREE_X']),float(desc['GAZE_DEGREE_Y'])) # This is the mandatory label information

            if (temp_dir != next_dir):
                # Beginning of new clip.
                # Store the clip to clips
                self.clips.append(ImageClip(temp_dir,temp_img_filenames, temp_skinlessRegionMask_filenames, temp_regionMask_filenames, temp_eye_locations, temp_pupil_sizes, temp_iris_sizes, temp_eye_opennesses, temp_eye_ids))
                # Initialize for new clip
                temp_dir = next_dir
                temp_img_filenames = list()
                temp_regionMask_filenames = list()
                temp_skinlessRegionMask_filenames = list()
                temp_eye_locations = list()
                temp_pupil_sizes = list()
                temp_iris_sizes = list()
                temp_eye_opennesses = list()
                temp_eye_ids = list()
            
            # accumulate filenames and other labels
            temp_img_filenames.append(desc['IMG_FILENAME'])
            temp_eye_ids.append(desc['EYE'])
            if 'SKINLESS_REGION_MASK_FILENAME' in desc:
                temp_skinlessRegionMask_filenames.append(desc['SKINLESS_REGION_MASK_FILENAME'])
            if 'REGION_MASK_FILENAME' in desc:
                temp_regionMask_filenames.append(desc['REGION_MASK_FILENAME'])
            if 'SHIFTED_EYE_X' in desc:
                temp_eye_locations.append((float(desc['SHIFTED_EYE_X']),float(desc['SHIFTED_EYE_Y']),float(desc['SHIFTED_EYE_Z'])))
            # pdb.set_trace()
            elif 'SLIPPAGE_X' in desc:
                # temp_eye_locations.append(
                #     (
                #         float(desc['ORIGINAL_EYE_X']) + float(desc['SLIPPAGE_X']),
                #         float(desc['ORIGINAL_EYE_Y']) + float(desc['SLIPPAGE_Y']),
                #         float(desc['ORIGINAL_EYE_Z']) + float(desc['SLIPPAGE_Z']),
                #     )
                # )
                temp_eye_locations.append(
                    (
                        float(desc['SLIPPAGE_X']),
                        float(desc['SLIPPAGE_Y']),
                        float(desc['SLIPPAGE_Z']),
                    )
                )
            temp_pupil_sizes.append(float(desc['PUPILSIZE']))
            if 'IRISSIZE' in desc:
                temp_iris_sizes.append(float(desc['IRISSIZE']))
            else:
                temp_iris_sizes.append(float(0)) # temporary fix for not having iris size in the label.
            temp_eye_opennesses.append(float(desc['EYEOPENNESS']))
            if (i == len(desc_list) - 1):
                # Beginning of new clip or the end of data.
                # Store the clip to clips
                self.clips.append(ImageClip(temp_dir,temp_img_filenames, temp_skinlessRegionMask_filenames, temp_regionMask_filenames, temp_eye_locations, temp_pupil_sizes, temp_iris_sizes, temp_eye_opennesses, temp_eye_ids))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx]

    def __iter__(self):
        return (clip for clip in self.clips)

#----------------------------------------------------------------------------

def resize_array(img, size):
    # Removing the following two lines. Previously we only dealt with gray images, so more than one color channel means we need to convert the given image to a gray image. But now we also have masks generated using all three color channels. The experimenter has to be responsible for keeping color channel to be one for any gray images.
    # if len(img.shape) > 2:
    #     img = img[:,:,0]

    if len(img.shape) > 2: # color image
        img = PIL.Image.fromarray(img, mode = 'RGB')
    else: # gray
        img = PIL.Image.fromarray(img)
    img = img.resize(size, PIL.Image.BICUBIC)
    img = np.clip(img, 0.0, 255.5)
    img = np.asarray(img, dtype=np.uint8)
    return img

def resize_region_map(region_map, size):
    # Similar to resize_array, but 2 differences.
    # 1. Nearest neighbor sampling to keep the region identification.
    # 2. No clipping to [0, 255]
    region_map = cv2.resize(region_map, dsize = (size[0],size[1]), interpolation = cv2.INTER_NEAREST)
    return region_map

#----------------------------------------------------------------------------

def choose_test_clips(clip_dir, subsample):
    unique_clip_dir = np.unique(clip_dir, axis = 0)
    widthRange = (min(unique_clip_dir, key=lambda x : x[0])[0], max(unique_clip_dir, key=lambda x : x[0])[0])
    heightRange = (min(unique_clip_dir, key=lambda x : x[1])[1], max(unique_clip_dir, key=lambda x : x[1])[1])
    binSize = math.sqrt((widthRange[1] - widthRange[0]) * (heightRange[1] - heightRange[0]) / (len(unique_clip_dir) / float(subsample))) # the area of bin is: binSize * binSize
    binDimension = (widthRange[1] // binSize - widthRange[0] // binSize + 1, heightRange[1] // binSize - heightRange[0] // binSize + 1)
    binIndex = [int(binDimension[0]*((x[1] - heightRange[0]) // binSize) + ((x[0] - widthRange[0]) // binSize)) for x in unique_clip_dir]
    train_or_test = [0] * len(unique_clip_dir)

    TIME_mod = 0
    TIME_is_in_this_bin = 0
    TIME_cumsum = 0
    TIME_binmask = 0
    TIME_rand = 0
    TIME_train_or_test = 0
    # print('Creating a test set. Going through ' + str(binSize) + ' sets...' + '\n')
    # for i in range(binSize):
    #     print('   Selecting a sample direction for bin number ' + str(i) + '\n')

    print('    Selection began...\n')
    count_per_bin = [0] * (max(binIndex) + 1) # a data structure for sample selection in a single pass
    selected_per_bin = [float('nan')] * (max(binIndex) + 1)
    # For unknown size of bin, we can guarantee to do a random, fair selection procedure by applying the following probability to each new sample we encounter.
    # 1 for the first sample
    # (N-1) / N for the Nth sample
    # For this we should keep track of the count of samples we have encountered so far, and we have to do that per each bin.
    for i,b_i in enumerate(binIndex):
        count_per_bin[b_i] += 1
        if count_per_bin[b_i] == 1:
            probability = 1
        else:
            probability = (count_per_bin[b_i] - 1) / count_per_bin[b_i]
        if random.uniform(0,1) > probability:
            selected_per_bin[b_i] = i

    print('    Selection list generated...\n')
    # Now alter train_or_test for the selcted indices.
    for s in selected_per_bin:
        if not math.isnan(s):
            train_or_test[s] = 1

    # This train_or_test is for unique_clip_dir. Generate train_or_test mask for the original clip_dir variable
    full_train_or_test = np.asarray([0] * len(clip_dir))
    selected_dir = unique_clip_dir[np.asarray(train_or_test) == 1]
    for ii in range(clip_dir.shape[0]):
        if clip_dir[ii] in selected_dir:
            full_train_or_test[ii] = 1
    train_or_test = full_train_or_test

    print('    Train mask generated...\n')
    return train_or_test

#----------------------------------------------------------------------------
def generateRegionMap(region_mask, skinless_region_mask, output_resolution = (255,191)):
    # get aperture mask
    # for now we only have one setting: GearVR-based on-axis tracking. Get the aperture mask generated for that particular setting. This function will grow as we have more settings.
    if platform.system() == 'Linux':
        mask_img = PIL.Image.open('/playpen/data/data_etc/ValidArea_%dx%d.png'%(output_resolution[0],output_resolution[1])).convert('L')
    elif platform.system() == 'Windows':
        drive = os.path.abspath(os.sep)
        mask_img = PIL.Image.open(os.path.join(drive, 'data_etc', 'ValidArea_%dx%d.png'%(output_resolution[0],output_resolution[1]))).convert('L')
        
    aperture_mask = np.round(np.array(mask_img) / 255.0).astype(bool)

    # Values in region map
    pupil_value = 1
    iris_value = 2
    sclera_value = 3
    skin_value = 10
    glint1_value = 100
    glint2_value = 200
    glint3_value = 300
    glint4_value = 400
    glint5_value = 500
    # A few examples on how to interpret the mask.
    # Pupil unoccluded by anything: 1 (pupil)
    # Pupil occluded by skin: 11 = 10 (skin) + 1 (pupil)
    # Iris occluded by skin: 12 = 10 (skin) + 2 (iris)
    # Sclera occluded by skin: 13 = 10 (skin) + 3 (sclera)
    # Pupil with glint on it: 101 = 100 (glint) + 1 (pupil)
    rm0 = region_mask[:,:,0]
    rm1 = region_mask[:,:,1]
    rm2 = region_mask[:,:,2]
    sm0 = skinless_region_mask[:,:,0]
    sm1 = skinless_region_mask[:,:,1]
    sm2 = skinless_region_mask[:,:,2]

    sclera_mask = np.logical_and(sm0>0, aperture_mask)
    iris_mask = np.logical_and(sm1 > 205, aperture_mask)
    pupil_mask = np.logical_and(np.logical_and(sm0 < 100, sm1 > 100), aperture_mask)
    sclera_mask[np.logical_or(np.logical_and(iris_mask,sclera_mask),np.logical_and(pupil_mask,sclera_mask))] = 0 # remove iris and pupil region from sclera
    iris_mask[np.logical_and(pupil_mask,iris_mask)] = 0 # remove pupil region from iris mask

    # pupil_mask = np.logical_and(sm0 < 100, sm1 > 100)
    # iris_mask = np.logical_and(sm0==255, sm1 > 205)
    # iris_mask[np.logical_and(pupil_mask,iris_mask)] = 0 # make pupil_mask and iris_mask mutually exclusive.
    # sclera_mask = np.logical_and(sm0==255, sm1 < 205, sm1 > 190)
    # sclera_mask[np.logical_and(sclera_mask,iris_mask)] = 0 # make sclera_mask and iris_mask mutually exclusive.

    skin_mask = np.logical_and(np.logical_and(rm0>200, rm1==0), rm2==0) # don't need to multiply aperture mask because we don't care about its position too much and having some extra extent over the aperture mask helps when blurring skin area during augmentation.

    glint1_mask = np.logical_and(np.logical_and(np.logical_and(rm0 < 250, rm1 == 255), rm2 < 250), aperture_mask)
    glint2_mask = np.logical_and(np.logical_and(np.logical_and(rm0 < 250, rm1 < 250), rm2 == 255), aperture_mask)
    glint3_mask = np.logical_and(np.logical_and(np.logical_and(rm0 == 255, rm1 == 255), rm2 < 250), aperture_mask)
    glint4_mask = np.logical_and(np.logical_and(np.logical_and(rm0 == 255, rm1 < 250), rm2 == 255), aperture_mask)
    glint5_mask = np.logical_and(np.logical_and(np.logical_and(rm0 < 250, rm1 == 255), rm2 == 255), aperture_mask)

    pupil_map = np.zeros(pupil_mask.shape)
    pupil_map[pupil_mask] = pupil_value
    iris_map = np.zeros(iris_mask.shape)
    iris_map[iris_mask] = iris_value
    sclera_map = np.zeros(sclera_mask.shape)
    sclera_map[sclera_mask] = sclera_value
    skin_map = np.zeros(skin_mask.shape)
    skin_map[skin_mask] = skin_value
    glint1_map = np.zeros(glint1_mask.shape)
    glint1_map[glint1_mask] = glint1_value
    glint2_map = np.zeros(glint2_mask.shape)
    glint2_map[glint2_mask] = glint2_value
    glint3_map = np.zeros(glint3_mask.shape)
    glint3_map[glint3_mask] = glint3_value
    glint4_map = np.zeros(glint4_mask.shape)
    glint4_map[glint4_mask] = glint4_value
    glint5_map = np.zeros(glint5_mask.shape)
    glint5_map[glint5_mask] = glint5_value
    regionMap = pupil_map + iris_map + sclera_map + skin_map + glint1_map + glint2_map + glint3_map + glint4_map + glint5_map
    # im = PIL.Image.fromarray(np.uint8(regionMap))
    # im.save('region_map.png')
    return regionMap

#----------------------------------------------------------------------------

def export_h5(dataset,out_dir, args, output_resolution = (255, 191)):
    subsample = args.subsample
    # Target resolution for images.
    
    print("Dataset contains %d clips." % len(dataset))
    # output_resolution = (320, 240) # (width, height)

    # Load good frame counts.

    good_frames = dict()
    try:
        with codecs.open(os.path.join(dataset.path, 'good_frame_counts.txt'), 'rt', encoding="utf-8-sig") as f:
            for line in f:
                s = line.strip().split(',')
                good_frames[s[0]] = int(s[1])
    except:
        print("No good_frame_counts.txt in dataset directory, using all frames.")
        good_frames = None

    # Create clip index.
    clip_dir = [clip.dir for clip in dataset]
    clip_dir = np.asarray(clip_dir, dtype=np.float32)
    clip_range = []
    total = 0
    for clip in dataset:
        cnt = (good_frames[os.path.splitext(os.path.basename(clip.path))[0]]) if (good_frames is not None) else len(clip)
        clip_range.append((total, cnt))
        total += cnt
    clip_range = np.asarray(clip_range, dtype=np.int32)
    print("In total we have %d good frames." % total)
    print("clip_dir.shape = %s" % str(clip_dir.shape))

    # 1. Generate 'clip_range' and organize them for train and test set if necessary. When 'subsample' is defined, we will make a test set by picking 1 out of every 'subsample' samples and a train set will be the rest.
    h5_clip_infos = list()
    if subsample is None:
        h5_clip_infos.append({'set_name':'all','clip_range':clip_range,'clip_dir':clip_dir,'total':total})
        train_or_test = [0] * len(clip_dir)
    elif subsample is not None:
        h5_clip_infos.append({'set_name':'train','clip_range':list(),'clip_dir':list(),'total':0})
        h5_clip_infos.append({'set_name':'test' ,'clip_range':list(),'clip_dir':list(),'total':0})
        train_or_test = choose_test_clips(clip_dir,subsample) # determine if each clip will be train (0) or test (1)
        for idx, tt in enumerate(train_or_test):
            h5_clip_infos[tt]['clip_dir'].append(clip_dir[idx])
            if len(h5_clip_infos[tt]['clip_range']) == 0:
                h5_clip_infos[tt]['clip_range'].append([0,clip_range[idx][1]])
            else:
                h5_clip_infos[tt]['clip_range'].append([h5_clip_infos[tt]['clip_range'][-1][0] + h5_clip_infos[tt]['clip_range'][-1][1],clip_range[idx][1]])
        for ii in range(len(h5_clip_infos)):
            h5_clip_infos[ii]['clip_dir'] = np.asarray(h5_clip_infos[ii]['clip_dir'])
            h5_clip_infos[ii]['clip_range'] = np.asarray(h5_clip_infos[ii]['clip_range'])
            #train_cnt = [clip_range[i][1] if t == 0 else 0 for i,t in enumerate(train_or_test)] # count number of train images
            h5_clip_infos[ii]['total'] = np.sum(h5_clip_infos[ii]['clip_range'][:,1])
    else:
        print("ERROR: Unhandled case!\n")

    # 2. Initialize h5 file generators
    h5_data = list()
    f = list()
    for ii, h5i in enumerate(h5_clip_infos):
        print('h5 file generation progress %d/%d...'%(ii,len(h5_clip_infos)))
        out_fn = os.path.join(out_dir,os.path.basename(dataset.path) + '_' + h5i['set_name'] + '_' + args.eye + '_' + str(output_resolution[0]) + 'x' + str(output_resolution[1]) + '.h5')
        print("Exporting to '%s'." % out_fn)
        f.append(h5py.File(out_fn, 'w'))
        # directly assing values if they are already known.
        f[-1].create_dataset('clip_dir', h5i['clip_dir'].shape, h5i['clip_dir'].dtype)[...] = h5i['clip_dir']
        f[-1].create_dataset('clip_range', h5i['clip_range'].shape, h5i['clip_range'].dtype)[...] = h5i['clip_range']
        # allocating seats for the values to be read from the dataset
        dummy1, dummy2, dummy3 = dataset.clips[0].get_image_and_mask(0)
        if len(dummy2.shape) == 0: # if mask doesn't exist
            h5_data.append({
                'images':f[-1].create_dataset('images', (h5i['total'], output_resolution[1], output_resolution[0]), np.uint8),
                'directions':f[-1].create_dataset('labels/directions', (h5i['total'], 2), np.float32),
                'subjects':f[-1].create_dataset('labels/subjects', (h5i['total'], ), np.float32),
                'eye_ids':f[-1].create_dataset('labels/eye_ids', (h5i['total'], ), np.float32),
            })
        else:
            h5_data.append({
                'images':f[-1].create_dataset('images', (h5i['total'], output_resolution[1], output_resolution[0]), np.uint8),
                'directions':f[-1].create_dataset('labels/directions', (h5i['total'], 2), np.float32),
                'subjects':f[-1].create_dataset('labels/subjects', (h5i['total'], ), np.float32),
                'eye_ids':f[-1].create_dataset('labels/eye_ids', (h5i['total'], ), np.float32),
                'eye_locations':f[-1].create_dataset('labels/eye_locations', (h5i['total'], 3), np.float32),
                'pupil_sizes':f[-1].create_dataset('labels/pupil_sizes', (h5i['total'], ), np.float32),
                'iris_sizes':f[-1].create_dataset('labels/iris_sizes', (h5i['total'], ), np.float32),
                'eye_opennesses':f[-1].create_dataset('labels/eye_opennesses', (h5i['total'], ), np.float32),
                'unoccluded_pupil_centers':f[-1].create_dataset('labels/unoccluded_pupil_centers', (h5i['total'], 2), np.float32),
                'unoccluded_pupil_bounding_boxes':f[-1].create_dataset('labels/unoccluded_pupil_bounding_boxes', (h5i['total'], 4), np.float32),
                'unoccluded_pupil_ellipse_fits':f[-1].create_dataset('labels/unoccluded_pupil_ellipse_fits', (h5i['total'], 5), np.float32),
                'unoccluded_iris_centers':f[-1].create_dataset('labels/unoccluded_iris_centers', (h5i['total'], 2), np.float32),
                'unoccluded_iris_bounding_boxes':f[-1].create_dataset('labels/unoccluded_iris_bounding_boxes', (h5i['total'], 4), np.float32),
                'unoccluded_iris_ellipse_fits':f[-1].create_dataset('labels/unoccluded_iris_ellipse_fits', (h5i['total'], 5), np.float32),
                'unoccluded_pupil_to_iris_ratios':f[-1].create_dataset('labels/unoccluded_pupil_to_iris_ratios', (h5i['total'], ), np.float32),
                'pupil_occlusion_ratios':f[-1].create_dataset('labels/pupil_occlusion_ratios', (h5i['total'], ), np.float32),
                'iris_occlusion_ratios':f[-1].create_dataset('labels/iris_occlusion_ratios', (h5i['total'], ), np.float32),
                'glint1_centers':f[-1].create_dataset('labels/glint1_centers', (h5i['total'], 2), np.float32),
                'glint2_centers':f[-1].create_dataset('labels/glint2_centers', (h5i['total'], 2), np.float32),
                'glint3_centers':f[-1].create_dataset('labels/glint3_centers', (h5i['total'], 2), np.float32),
                'glint4_centers':f[-1].create_dataset('labels/glint4_centers', (h5i['total'], 2), np.float32),
                'glint5_centers':f[-1].create_dataset('labels/glint5_centers', (h5i['total'], 2), np.float32),
                'glint_counts':f[-1].create_dataset('labels/glint_counts', (h5i['total'], ), np.float32),
                'region_maps':f[-1].create_dataset('region_maps', (h5i['total'], output_resolution[1], output_resolution[0]), np.float32),
            })

    # 3. dataset is an iterator. Going through each clip, copy information to the right h5_data.
    # Use of temp_start_index could also be avoided. Will remove that part during the next implementation phase.
    h5_start_indices = [0] * len(h5_data)
    for i, clip in enumerate(dataset):
        # full report of progress per every frame
        # print("Converting clip %d / %d" % (i+1,len(dataset)))

        # shortened report of progress per every 10 %
        progress_rate_to_report = np.arange(0,1,0.1)
        clip_index_for_reporting_progress = (len(dataset) * progress_rate_to_report).astype(np.int32)
        if any(i == clip_index_for_reporting_progress):
            print("    Conversion progress: %d percent done." % int(progress_rate_to_report[i == clip_index_for_reporting_progress][0] * 100))
        cnt = clip_range[i][1]
        which_h5 = train_or_test[i]
        imgs = []
        region_maps = []
        unoccluded_pupil_centers = []
        unoccluded_pupil_bounding_boxes = []
        unoccluded_pupil_ellipse_fits = []
        unoccluded_iris_centers = []
        unoccluded_iris_bounding_boxes = []
        unoccluded_iris_ellipse_fits = []
        unoccluded_pupil_to_iris_ratios = []
        pupil_occlusion_ratios = []
        iris_occlusion_ratios = []
        glint1_centers = []
        glint2_centers = []
        glint3_centers = []
        glint4_centers = []
        glint5_centers = []
        glint_counts = []
        for j in range(cnt):
            img, region_mask, skinless_region_mask = clip[j]
            if len(img.shape) == 3: # strip off the channel dimension
                img = np.squeeze(img[:,:,0])
            imgs.append(resize_array(img, output_resolution))
            if len(region_mask.shape) > 0: # We have region masks. Generate region map and pupil center info.
                glint_count = 0
                region_mask = resize_region_map(region_mask, output_resolution)
                skinless_region_mask = resize_region_map(skinless_region_mask, output_resolution)
                region_map = generateRegionMap(region_mask, skinless_region_mask, output_resolution)
                region_maps.append(region_map)
                # image pixel addresses
                xv, yv = np.meshgrid(np.arange(0,region_map.shape[1],1),np.arange(region_map.shape[0]-1,-1,-1))
                unoccluded_pupil_mask = np.mod(region_map,10) == 1
                open_pupil_mask = np.mod(region_map,100) == 1
                unoccluded_iris_mask = np.logical_or(np.mod(region_map,10) == 2, unoccluded_pupil_mask)
                open_iris_mask = np.logical_or(np.mod(region_map,100) == 2, open_pupil_mask)
                unoccluded_pupil_centers.append((
                    np.sum(yv * unoccluded_pupil_mask)/np.sum(unoccluded_pupil_mask),
                    np.sum(xv * unoccluded_pupil_mask)/np.sum(unoccluded_pupil_mask)
                ))
                pupil_rows, pupil_cols = np.nonzero(unoccluded_pupil_mask)
                unoccluded_pupil_bounding_boxes.append((pupil_cols.min(), pupil_rows.min(), pupil_cols.max(), pupil_rows.max()))
                # fit an ellipse to the pupil
                pupil_img = unoccluded_pupil_mask.astype(np.uint8) * 255
                _, contours, _ = cv2.findContours(pupil_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # choose the contour with the most number of elements.
                c_idx = 0
                c_size = 0
                for contour_idx, contour in enumerate(contours):
                    if contour.size > c_size:
                        c_idx = contour_idx
                        c_size = contour.size
                # if i == 6034:
                #     pdb.set_trace()
                try:
                    pupil_pos, pupil_axis, pupil_angle = cv2.fitEllipse(contours[c_idx])
                    unoccluded_pupil_ellipse_fits.append((pupil_pos[0], pupil_pos[1], pupil_axis[0], pupil_axis[1], pupil_angle))
                except:
                    pdb.set_trace()

                unoccluded_iris_centers.append((
                    np.sum(yv * unoccluded_iris_mask)/np.sum(unoccluded_iris_mask),
                    np.sum(xv * unoccluded_iris_mask)/np.sum(unoccluded_iris_mask)
                ))
                iris_rows, iris_cols = np.nonzero(unoccluded_iris_mask)
                unoccluded_iris_bounding_boxes.append((iris_cols.min(), iris_rows.min(), iris_cols.max(), iris_rows.max()))
                unoccluded_pupil_to_iris_ratios.append(np.sum(unoccluded_pupil_mask)/np.sum(unoccluded_iris_mask))
                # fit an ellipse to the iris
                iris_img = unoccluded_iris_mask.astype(np.uint8) * 255
                _, contours, _ = cv2.findContours(iris_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # choose the contour with the most number of elements.
                c_idx = 0
                c_size = 0
                for contour_idx, contour in enumerate(contours):
                    if contour.size > c_size:
                        c_idx = contour_idx
                        c_size = contour.size
                # if i == 6034:
                #     pdb.set_trace()
                try:
                    iris_pos, iris_axis, iris_angle = cv2.fitEllipse(contours[c_idx])
                    unoccluded_iris_ellipse_fits.append((iris_pos[0], iris_pos[1], iris_axis[0], iris_axis[1], iris_angle))
                except:
                    pdb.set_trace()

                pupil_occlusion_ratios.append(np.sum(open_pupil_mask)/np.sum(unoccluded_pupil_mask))
                iris_occlusion_ratios.append(np.sum(open_iris_mask)/np.sum(unoccluded_iris_mask))
                for glint_index in np.arange(1,6):
                    glint_mask = np.round(region_map/100) == glint_index
                    if np.sum(glint_mask) == 0:
                        center = (0,0) # (0,0) is not possible because of aperture limitation. Meaning that it doesn't exist.
                    else:
                        center = (
                            np.sum(yv * glint_mask)/np.sum(glint_mask),
                            np.sum(xv * glint_mask)/np.sum(glint_mask)
                        )
                        glint_count += 1
                    exec('glint%d_centers.append(center)'%glint_index)
                glint_counts.append(glint_count)
        h5_data[which_h5]['images'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.stack(imgs)
        if len(region_maps)>0:
            h5_data[which_h5]['region_maps'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.stack(region_maps)
            h5_data[which_h5]['unoccluded_pupil_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_pupil_centers, dtype = np.float32)
            h5_data[which_h5]['unoccluded_pupil_bounding_boxes'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_pupil_bounding_boxes, dtype = np.float32)
            h5_data[which_h5]['unoccluded_pupil_ellipse_fits'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_pupil_ellipse_fits, dtype = np.float32)
            h5_data[which_h5]['unoccluded_iris_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_iris_centers, dtype = np.float32)
            h5_data[which_h5]['unoccluded_iris_bounding_boxes'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_iris_bounding_boxes, dtype = np.float32)
            h5_data[which_h5]['unoccluded_iris_ellipse_fits'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_iris_ellipse_fits, dtype = np.float32)
            h5_data[which_h5]['unoccluded_pupil_to_iris_ratios'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(unoccluded_pupil_to_iris_ratios, dtype = np.float32)
            h5_data[which_h5]['glint1_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint1_centers, dtype = np.float32)
            h5_data[which_h5]['glint2_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint2_centers, dtype = np.float32)
            h5_data[which_h5]['glint3_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint3_centers, dtype = np.float32)
            h5_data[which_h5]['glint4_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint4_centers, dtype = np.float32)
            h5_data[which_h5]['glint5_centers'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint5_centers, dtype = np.float32)
            h5_data[which_h5]['pupil_occlusion_ratios'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(pupil_occlusion_ratios, dtype = np.float32)
            h5_data[which_h5]['iris_occlusion_ratios'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(iris_occlusion_ratios, dtype = np.float32)
            h5_data[which_h5]['glint_counts'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(glint_counts, dtype = np.float32)
        h5_data[which_h5]['directions'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['directions'][:clip_range[i][1]], dtype = np.float32)
        h5_data[which_h5]['subjects'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['subjects'][:clip_range[i][1]], dtype = np.float32)
        eye_ids_string = clip.labels['eye_ids'][:clip_range[i][1]]
        # Storing string in h5py seems a bit inconvenient (encoding conversion necessary). Convert ('L' or 'R') to (0 or 1)
        eye_ids_float = [0 if eye_id == 'L' else 1 for eye_id in eye_ids_string]
        h5_data[which_h5]['eye_ids'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(eye_ids_float, dtype = np.float32)
        if 'eye_locations' in clip.labels: # If not, the clip is also missing pupil_sizes, iris_sizes, and eye_opennesses
            h5_data[which_h5]['eye_locations'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['eye_locations'][:clip_range[i][1]], dtype = np.float32)
            h5_data[which_h5]['pupil_sizes'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['pupil_sizes'][:clip_range[i][1]], dtype = np.float32)
            h5_data[which_h5]['iris_sizes'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['iris_sizes'][:clip_range[i][1]], dtype = np.float32)
            h5_data[which_h5]['eye_opennesses'][h5_start_indices[which_h5] : h5_start_indices[which_h5] + cnt, ...] = np.asarray(clip.labels['eye_opennesses'][:clip_range[i][1]], dtype = np.float32)
        h5_start_indices[which_h5] += cnt
    for each_f in f:
        each_f.close()

    print("Done.")

#----------------------------------------------------------------------------

def label_blinks(sequence, start, end):
    interp = interp1d(np.array((0,start,end,1000)), np.array((0.,0.,1.,1.)))
    files = os.listdir(sequence)
    files.sort()
    for i, f in enumerate(files):
        tok = f.split('_')
        print(type(i))
        print(interp.dtype)
        x = interp(i)
        print(x)
        tok[5] = '%s'%interp(int(i))
        print(tok)
        os.rename(os.path.join(sequence,f), os.path.join(sequence,'_'.join(tok)))

def label_zipped(zip, start, end):
    #extract zip
    with zipfile.ZipFile(zip, 'r') as mz:
        mz.extractall(zip[:-4])
    print('Labelling %s'%zip[:-4])
    #do the labelling
    label_blinks(zip[:-4], start, end)
    #zip up the labeled files
    os.remove(zip)
    with zipfile.ZipFile(zip, 'w') as mz:
        for f in os.listdir(zip[:-4]):
            mz.write(os.path.join(zip[:-4],f),f)
            os.remove(os.path.join(zip[:-4],f))
    os.rmdir(zip[:-4])

def label_zip_folder(folder, labelFile):
    with codecs.open(labelFile, "r", encoding="utf-8-sig") as f:
        for line in f:
            dirName, start, end = line.split(',')
            label_zipped(os.path.join(folder, '%s.zip'%dirName), start, end.strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--subsample',default = None, help = 'Subsample the gaze directions to generate the test set. If provided, it will generate train set and test set separately.')
    parser.add_argument('-e','--eye',default = 'left', help = "'left' or 'right' means monocular data for left or right eye. 'binocular' means binocular data.")
    parser.add_argument('-x','--x_resolution',default = 255, help = "Horizontal dimension of images")
    parser.add_argument('-y','--y_resolution',default = 191, help = "Vertical dimension of images")
    args = parser.parse_args()

    prep_setting = dict()
    prep_setting['subject'] = [
        # 'nxp_female_01','nxp_female_02','nxp_female_03','nxp_female_04','nxp_female_05',
        # 'nxp_male_01','nxp_male_02','nxp_male_03','nxp_male_04','nxp_male_05',
        'nxp_female_01','nxp_female_02','nxp_female_04',
    ]
    prep_setting['plan'] = [
        'Plan002',
        # 'TestRobustnessToBlink',
        # 'TestRobustnessToPupilSize',
        # 'TestRobustnessToSlippage',
        # 'TestRobustnessToTransversalSlippage',
        # 'TestRobustnessToDepthSlippage',
        # 'short',
    ]


    # Pass the path where all the zip files are stored. Simply stores the path in dataset.path
    if platform.system() == 'Linux':
        prep_setting['parent_dir'] = [
            '/playpen/data/data_blender_glasses',
        ]

        src_paths = list()
        export_paths = list()
        for x in itertools.product(*prep_setting.values()):
            S = dict(zip(prep_setting.keys(),x))
            src_paths.append(os.path.join(S['parent_dir'],S['subject'] + '_' + S['plan']))
            export_paths.append(S['parent_dir'])
        # src_paths = ['/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v2/AM/compressed']

        src_paths = [
            # '/playpen/data/data_real_glasses/michael_1_zip'
            '/playpen/data/scratch/sckim/blender_output/nxp_female_01_Plan002',
            '/playpen/data/scratch/sckim/blender_output/nxp_female_01_Plan002_no_glint',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink/compressed',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_pupil/compressed',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_horizontal/compressed',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_vertical/compressed',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_stare/compressed',
            ]
        export_paths = [
            # '/playpen/data/data_real_glasses',
            '/playpen/data/scratch/sckim/blender_output',
            '/playpen/data/scratch/sckim/blender_output',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_pupil',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_horizontal',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_vertical',
        #     '/playpen/data/data_GearVR_Topfoison/onaxis_gaze_v3/JK_stare',
            ]
    elif platform.system() == 'Windows':
        parent_dir = [
            'X:/data_blender_glasses',
        ]

        # src_paths = list()
        # export_paths = list()
        # for x in itertools.product(*prep_setting.values()):
        #     S = dict(zip(prep_setting.keys(),x))
        #     src_paths.append(os.path.join(S['parent_dir'],S['subject'] + '_' + S['plan']))
        #     export_paths.append(S['parent_dir'])

        src_paths = [
            # 'X:/data_blender_onaxis/RefEye_EyeOpennessFullRange_Slippage',
            # 'X:/data_blender_onaxis_mask/nxp_female_02_Plan001_GlintWhite',
            # 'X:/data_blender_onaxis_mask/nxp_female_01_SimplePlan_GlintColorCoded',
            # 'X:/data_blender_glasses/nxp_female_01_short',
            # 'D:/temp/debugging',
            # 'X:/data_blender_glasses/nxp_female_01_Plan002',
            # 'X:/data_blender_glasses/nxp_female_02_Plan002',
            # 'X:/data_desktop/gaze_20170311/compressed'
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v2/JSK/short'
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink/compressed',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_pupil/compressed',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_stare/compressed',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_horizontal/compressed',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_slippage_vertical/compressed',
            'X:/data_real_glasses/michael_1_zip',
           ]
        export_paths = [
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_onaxis_mask',
            # 'X:/data_blender_glasses'
            # 'D:/temp',
            # 'X:/data_desktop/gaze_20170311/'
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v2'
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
            # 'X:/data_GearVR_Topfoison/onaxis_gaze_v3/JK_blink',
            'X:/data_real_glasses',
            ]

    for src_path, export_path in zip(src_paths, export_paths):
        # Currently 3 types of dataset are supported.
        # Type 1. Only images. This was the earliest format and contains only gaze direction labels in the file name.
        # Type 2. Images and a csv file that contains a list of labels that not only include gaze direction but also more info such as pupil size, iris size, eye ball locations, etc.
        # Type 3. Images, a csv file and two mask image sets. This type contains everything that is available in Type 2 and in addition pixel-wise maps showing to which region the given pixel belongs to. First set of mask images is generated with every component in it - skin, eyelashes, sclera, iris, pupil, and glint. Second set of mask images is generated with only the eye without glints. The second set can be used for the ground truth for pupil location when it is occluded by the eyelid.
        if len(glob.glob(os.path.join(src_path, '*.csv'))) == 1: # If there is a .csv file, it could be Type 2 or 3. The class ImageClipDataset will try to detect which type it is by looking for mask images.
            dataset = ImageClipDataset(src_path, args.eye)
        else: # if there is no .csv file, then it is type 1 (ZipDataset).
            dataset = ZipDataset(src_path, args.eye)
        dataset.init() # Loads files and store clips of image(s) to dataset.clips
        output_resolution = [int(args.x_resolution), int(args.y_resolution)]
        export_h5(dataset,export_path, args, output_resolution)
