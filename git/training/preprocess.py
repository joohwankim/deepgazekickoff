import torch, platform
from PIL import Image
import numpy as np
import pdb
import os
from torch.autograd import Variable
import torch.nn.functional as F
import time

debugging = False
probabilityDrawingSample = 0 # cast the dice, draw sample images by this amount of chance

class Preprocessor():
    # This class preprocesses inputs and also applies augmentation to training data.
    # Function 'SetupPlan' receives configuration and sets up the plan for data augmentation.
    # Function 'Preprocess' receives input image and region_map and performs preprocessing and augmentation using them. Augmentation only happens if it is training data.
    # Other functions are utility functions useful for preprocessing. But they can also be used for custom augmentation that is not defined by 'SetupPlan' function.

    # Same functionality as BlurAndBlendNet, but implemented using functional.
    def gen_blending_masks_list(self, region_maps):
        # Region map. Values in the map are explained.
        # pupil_value = 1
        # iris_value = 2
        # sclera_value = 3
        # skin_value = 10
        # glint_value = 100
        # A few examples on how to interpret the mask.
        # Pupil unoccluded by anything: 1 (pupil)
        # Pupil occluded by skin: 11 = 10 (skin) + 1 (pupil)
        # Iris occluded by skin: 12 = 10 (skin) + 2 (iris)
        # Sclera occluded by skin: 13 = 10 (skin) + 3 (sclera)
        # Pupil with glint on it: 101 = 100 (glint) + 1 (pupil)

        # We convert region maps to blending masks.
        pupil_mask = self.apply_blur((region_maps == 1).float(), self.blur_radius_for_region_map)
        iris_mask = self.apply_blur((region_maps == 2).float(), self.blur_radius_for_region_map)
        sclera_mask = self.apply_blur((region_maps == 3).float(), self.blur_radius_for_region_map)
        skin_mask = self.apply_blur((torch.fmod(region_maps,100) > 9).float(), self.blur_radius_for_region_map) # all the region occluded by skin
        glint_mask = self.apply_blur((region_maps > 99).float(), self.blur_radius_for_region_map) # all the region where glint is visible
        self.blending_masks_list = list()
        self.blending_masks_list.append(pupil_mask)
        self.blending_masks_list.append(iris_mask)
        self.blending_masks_list.append(sclera_mask)
        self.blending_masks_list.append(skin_mask)
        self.blending_masks_list.append(glint_mask)
    
    def gen_blur_kernel(self,blur_radius, blur_kernel_size, channel_count):
        # Generates an arbtrarily deep blur kernels.
        # Apply blur with radius of blur_radius to all the channels.
        blur_radius = [blur_radius] * channel_count
            
        grid_vector = np.linspace(-(blur_kernel_size - 1) / 2,(blur_kernel_size - 1) / 2,blur_kernel_size,endpoint = True)
        xx, yy = np.meshgrid(grid_vector,grid_vector)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        blur_kernel = np.zeros((len(blur_radius),len(blur_radius),blur_kernel_size,blur_kernel_size))

        for b_index, this_blur_radius in enumerate(blur_radius):
            if this_blur_radius == 0.0:
                this_blur_kernel = np.zeros(rr.shape)
                this_blur_kernel[int((blur_kernel_size-1)/2),int((blur_kernel_size-1)/2)] = 1
    #             this_blur_kernel = Variable(torch.FloatTensor(np.expand_dims(np.expand_dims(this_blur_kernel,axis=0),axis=0)),volatile = True).cuda()
            else:
                # cylindrical blur with sub-pixel approximation
                # The sub-pixel approximation assumes each pixel area is a square facing the center of the circle.
                # If the center of the pixel is farther than (blur_radius + 0.5) from the center, then the pixel is dark.
                # If the center of the pixel is closer than (blur_radius - 0.5) from the center, then the pixel is white.
                # In between, the pixel value changes from dark to white linearly as a function of the distance of the pixel from the center.
                # First, calculate the inverse image of the kernel because that is easier to understand (at least to me...)
                this_blur_kernel = rr - this_blur_radius
                this_blur_kernel[rr < (this_blur_radius - 0.5)] = 0.0
                this_blur_kernel[rr > (this_blur_radius + 0.5)] = 1.0
                this_blur_kernel = 1.0 - this_blur_kernel # and then take the invers of it to get the correct blur kernel.
                this_blur_kernel = this_blur_kernel / np.sum(this_blur_kernel) # normalization
            
            # stack each blur kernel channel to the blur kernel
            blur_kernel[b_index,b_index,:] = this_blur_kernel
    #             blur_kernel = Variable(torch.FloatTensor(np.expand_dims(np.expand_dims(blur_kernel,axis=0),axis=0)), volatile = True).cuda()
        blur_kernel = Variable(torch.FloatTensor(blur_kernel),volatile = True).cuda()

        return blur_kernel

    def get_mask(self, input_resolution, input_path):
        # for now we only have one setting: GearVR-based on-axis tracking. This function returns a mask generated for that particular setting. This function will grow as we have more settings.
        if platform.system() == 'Linux':
            mask_img = Image.open('/playpen/data/data_etc/ValidArea_' + str(input_resolution[0]) + 'x' + str(input_resolution[1]) + '.png').convert('L')
        elif platform.system() == 'Windows':
            #drive = os.path.abspath(os.sep)
            
            mask_img = Image.open(os.path.join(input_path, 'data_etc', 'ValidArea_' + str(input_resolution[0]) + 'x' + str(input_resolution[1]) + '.png')).convert('L')
        
        mask = np.round(np.expand_dims(np.expand_dims(np.array(mask_img) / 255.0,axis=0),axis=0))
        # mask[mask == 0.0] = np.nan
        mask = Variable((mask * torch.ones(mask.shape)).float(), volatile = True).cuda()
        return mask

    def update_plan(self, config):
        self.config = config

    def __init__(self, config):
        # group_size is the number of images that are going to share the same augmentation parameters. This group will be transferred together to the gpu and augmentation will be processed. Preprocessing per each image would maximize the effectiveness of augmentation, but it does not exploit the computational efficiency of gpus. Larger group size will reduce the effectiveness but it will train faster.
        if hasattr(config, 'preprocessing_group_size'):
            self.group_size = config.preprocessing_group_size
        else:
            self.group_size = 20
        #self.aperture_mask = self.get_mask(config.input_resolution, config.data_dir)
        self.aperture_mask = self.get_mask(config.input_resolution, os.path.dirname(config.result_dir))
        self.config = config
        if 'binocular' in self.config.network_type.__name__.lower():
            self.channel_count = 2
            inv_idx3 = Variable(torch.arange(self.aperture_mask.size(3)-1, -1, -1).long()).cuda()
            self.aperture_mask_x_flipped = torch.index_select(self.aperture_mask,3,inv_idx3)
            self.aperture_mask = torch.stack([self.aperture_mask, self.aperture_mask_x_flipped],1).squeeze(2) # binocular version of aperture mask
        else:
            self.channel_count = 1
        self.blur_radius_for_region_map = 0.0
        if 'blur_radius_for_region_map' in config.augmentations:
            self.blur_radius_for_region_map = config.augmentations['blur_radius_for_region_map']
        self.blending_masks_list = None # To be generated out of region maps each time preprocessing is done.
        self.num_sample_store = 10 # if you want to check some sample images...

    def gen_blur_kernels(self, blur_radii, channel_count):
        padding_size = int(np.ceil(max(blur_radii)))
        blur_kernel_size = int(padding_size * 2 + 1)
        # blur kernels
        blur_kernels = [self.gen_blur_kernel(blur_radius, blur_kernel_size, channel_count) for blur_radius in blur_radii]
        return blur_kernels

    def update_contrast_modulation_amplitudes(self, contrast_modulation_amplitudes):
        self.contrast_modulation_amplitudes = contrast_modulation_amplitudes
            
    def apply_blur(self, inputs, blur_radius):
        # a utility function that blurs given input by the size of blur_radius
        # Applies the given amount of blur if blur_radius is greater than 0.0
        # inputs and region_maps should be in the same dimension
        if blur_radius > 0:
            padding_size = int(np.ceil(blur_radius))
            blur_kernel_size = int(padding_size * 2 + 1)
            blur_kernel = self.gen_blur_kernel(blur_radius, blur_kernel_size, inputs.shape[1])
            outputs = inputs.clone()
            outputs.data.zero_()
            outputs = outputs + F.conv2d(F.pad(inputs,(padding_size,)*4), blur_kernel)
        else:
            outputs = inputs
        return outputs

    def apply_region_wise_blur(self, inputs, region_maps, blur_radii, blur_radius_for_region_map = 0.0):
        # if blur radii is not given, use the blur defined the most recently.
        # inputs: 4D tensor (NCHW format)
        # region_maps: indicates pupil, iris, sclera, skin, and any occlusion by skin.
        # blur_radii (default None): Array of blur radius, length equal to the number of regions defined in region map. Updates blur kernel if given, uses the most recent values if not given.
        # blur_radius_for_region_map (default 0.0): takes positive rational number and will blur the region map by the given amount if greater than 0.

        # inputs and region_maps should be in the same dimension
        # temp = time.time()
        outputs = inputs.clone()
        # print('  Region-wise blur part 1: %.5f'%(time.time() - temp))

        # temp = time.time()
        outputs.data.zero_()
        # print('  Region-wise blur part 2: %.5f'%(time.time() - temp))

        # temp = time.time()
        padding_size = int(np.ceil(max(blur_radii)))
        # print('  Region-wise blur part 3: %.5f'%(time.time() - temp))

        # temp = time.time()
        blur_kernels = self.gen_blur_kernels(blur_radii, inputs.shape[1])
        # print('  Region-wise blur part 4: %.5f'%(time.time() - temp))

        # temp = time.time()
        for blur_kernel, blending_masks in zip(blur_kernels, self.blending_masks_list):
            outputs = outputs + F.conv2d(F.pad(inputs * blending_masks, (padding_size,)*4), blur_kernel)
        # print('  Region-wise blur part 5: %.5f'%(time.time() - temp))

        # temp = time.time()
        outputs = torch.clamp(outputs, 0, 255)
        # print('  Region-wise blur part 6: %.5f'%(time.time() - temp))
        return outputs

    # The following doesn't work for masked-region-based contrast modulation because mean should be calculated for only the masked regions.
    # def apply_contrast_modulation(self, inputs, contrast_modulation_amplitude):
    #     outputs = (inputs - inputs.mean()) * contrast_modulation_amplitude + inputs.mean()
    #     return outputs

    def apply_region_wise_contrast_modulation(self, inputs, region_maps, contrast_modulation_amplitudes, blur_radius_for_region_map = 0.0):
        # if blur radii is not given, use the blur defined the most recently.
        # inputs: 4D tensor (NCHW format)
        # region_maps: indicates pupil, iris, sclera, skin, and any occlusion by skin.
        # contrast_modulation_amplitudes: Array of contrast modulation magnitudes, length equal to the number of regions defined in region map.
        # blur_radius_for_region_map (default 0.0): takes positive rational number and will blur the region map by the given amount if greater than 0.
        # inputs and region_maps should be in the same dimension
        # temp = time.time()
        modulated_inputs = inputs.clone()
        modulated_inputs.data.zero_()
        # print('  Region-wise contrast normalization part 1: %.5f'%(time.time() - temp))

        # temp = time.time()
        for region_i in range(len(self.blending_masks_list)):
            contrast_modulation_amplitude = contrast_modulation_amplitudes[region_i]
            blending_mask = self.blending_masks_list[region_i]
            if blending_mask.sum().data.cpu().numpy() == 0:
                region_mean = 0
            else:
                region_mean = (inputs*blending_mask).sum() / blending_mask.sum()
            modulated_layer = (region_mean + (inputs - region_mean) * contrast_modulation_amplitude) * blending_mask
            modulated_inputs = modulated_inputs + modulated_layer
        # print('  Region-wise contrast normalization part 2: %.5f'%(time.time() - temp))

        # temp = time.time()
        inputs = torch.clamp(modulated_inputs, 0, 255)
        # print('  Region-wise contrast normalization part 3: %.5f'%(time.time() - temp))
        return inputs

    def normalize_with_contrast_modulation(self, inputs, global_contrast_perturbation):
        # calculate mean, max, and min of the source image in the valid area.
        # temp = time.time()
        # s_mean = torch.sum((inputs * self.aperture_mask).data) / (torch.sum(self.aperture_mask) * inputs.size(0))
        # s_max = torch.max((inputs * self.aperture_mask).data)
        # s_min = torch.min(((inputs * self.aperture_mask) + (1 - self.aperture_mask) * 255).data)
        # print('  Global contrast normalization part 1a: %.5f'%(time.time() - temp))

        # temp = time.time()
        s_mean = (inputs * self.aperture_mask).mean(3).mean(2)
        s_max = (inputs * self.aperture_mask).max(3)[0].max(2)[0]
        s_min = ((inputs * self.aperture_mask) + (1 - self.aperture_mask) * 255).min(3)[0].min(2)[0]
        # print('  Global contrast normalization part 1: %.5f'%(time.time() - temp))
        
        # calculate mean, max, and min of the target image in the valid area.
        # temp = time.time()
        t_mean = (np.random.random(s_mean.shape) * global_contrast_perturbation - global_contrast_perturbation / 2.0)
        t_max = 1.0
        t_min = -1.0
        # print('  Global contrast normalization part 2: %.5f'%(time.time() - temp))

        # contrast modulation amplitudes
        # temp = time.time()
        m_high = 1 + global_contrast_perturbation * (np.random.random(s_mean.shape) * 2 - 1) # for the part greater than the mean
        m_low = 1 + global_contrast_perturbation * (np.random.random(s_mean.shape) * 2 - 1) # for the part less than the mean
        # print('  Global contrast normalization part 3: %.5f'%(time.time() - temp))

        # temp = time.time()
        s_mean = s_mean.unsqueeze(2).unsqueeze(3)
        s_min = s_min.unsqueeze(2).unsqueeze(3)
        s_max = s_max.unsqueeze(2).unsqueeze(3)
        m_high = Variable(torch.FloatTensor(m_high), volatile = True, requires_grad = False).cuda().unsqueeze(2).unsqueeze(3)
        m_low = Variable(torch.FloatTensor(m_low), volatile = True, requires_grad = False).cuda().unsqueeze(2).unsqueeze(3)
        normalized_input_high = (inputs - s_mean) * (inputs > s_mean).float() / (s_max - s_mean) * m_high
        normalized_input_low =  (inputs - s_mean) * (inputs <= s_mean).float() / (s_mean - s_min) * m_low
        # print('  Global contrast normalization part 4: %.5f'%(time.time() - temp))

        # temp = time.time()
        inputs.data = torch.clamp(normalized_input_high + normalized_input_low, t_min, t_max).data
        # print('  Global contrast normalization part 5: %.5f'%(time.time() - temp))
        return inputs

    def apply_region_wise_intensity_offset(self, inputs, region_maps, intensity_offsets, blur_radius_for_region_map = 0.0):
        # if blur radii is not given, use the blur defined the most recently.
        # inputs: 4D tensor (NCHW format)
        # region_maps: indicates pupil, iris, sclera, skin, and any occlusion by skin.
        # contrast_modulation_amplitudes: Array of contrast modulation magnitudes, length equal to the number of regions defined in region map.
        # blur_radius_for_region_map (default 0.0): takes positive rational number and will blur the region map by the given amount if greater than 0.
        # inputs and region_maps should be in the same dimension
        for region_i in range(len(self.blending_masks_list)):
            offset = intensity_offsets[region_i]
            blending_masks = self.blending_masks_list[region_i]
            inputs = inputs + float(offset) * blending_masks
        inputs = torch.clamp(inputs, 0, 255)
        return inputs

    def apply_global_gaussian_noise(self,inputs, g_sigma):
        if g_sigma > 0:
            gaussian_noise = inputs.clone()
            gaussian_noise.data.normal_(0,g_sigma)
            inputs = inputs + gaussian_noise
        inputs = torch.clamp(inputs, 0, 255)
        return inputs

    def preprocess(self,inputs, region_maps, labels, do_augmentation = False):
        # if region_maps is not None:
        #     #print(region_maps)
        #     if len(region_maps) == 0:
        #     #if region_maps == None:
        #         #print("INFO : no region maps existing!")
        #         region_maps = None

        # Don't do augmentation if region_maps is None.
        outputs = inputs.clone()
        outputs.data.zero_()
        if self.config.network_type.__name__ == 'ResNet50' or self.config.network_type.__name__ == 'ResNet50FeaturesFrozen': # run preprocessing that's specific to ResNet50
            outputs = outputs.repeat(1,3,1,1)
        drawSample = False
        # if np.random.random(1) < probabilityDrawingSample:
        #     drawSample = True
        #     print('---------------drawing sample image!-------------------')

        for ii in range(int(np.ceil(inputs.size(0) / self.group_size))):
            i_from = int(self.group_size * ii)
            i_to = int(np.min([inputs.size(0),self.group_size*(ii + 1)]))
            input = inputs[i_from:i_to].clone() # doing this to maintain NCHW format
            if drawSample:
                img = Image.fromarray(np.squeeze(input[0].data.cpu().numpy()))
                img = img.convert('RGB')
                img.save('preprocess0_input_img.png')

            if len(region_maps.shape) == 4:
                region_map = region_maps[i_from:i_to] # doing this to maintain NCHW format
                self.gen_blending_masks_list(region_map)

            if (do_augmentation) and (len(region_maps.shape) == 4):
                region_num = len(self.config.augmentations['raw_intensity_offsets_perturbations'])
                block_start_time = time.time()
                if 'raw_intensity_offsets' in self.config.augmentations:
                    params = np.asarray(self.config.augmentations['raw_intensity_offsets'])
                    if 'raw_intensity_offsets_perturbations' in self.config.augmentations:
                        params = params + np.asarray(self.config.augmentations['raw_intensity_offsets_perturbations']) * (2 * np.random.random(region_num) -1)
                    # if debugging: print('Raw intensity offset parameters: %f, %f, %f, %f'%(params[0],params[1],params[2],params[3],))
                    input = self.apply_region_wise_intensity_offset(input, region_map, params, self.blur_radius_for_region_map)
                    # if debugging: print('After intensity offset, min: %f max: %f'%(input.min(),input.max()))
                    if drawSample:
                        img = Image.fromarray(np.squeeze(input[0].data.cpu().numpy()))
                        img = img.convert('RGB')
                        img.save('preprocess1_offset_added_img.png')
                if debugging: print('intensity offset takes: %.5f'%(time.time() - block_start_time))

                block_start_time = time.time()
                if 'contrast_scalings' in self.config.augmentations:
                    params = np.asarray(self.config.augmentations['contrast_scalings'])
                    if 'contrast_scalings_perturbations' in self.config.augmentations:
                        params = params + np.asarray(self.config.augmentations['contrast_scalings_perturbations']) * (2 * np.random.random(region_num) -1)
                    # if debugging: print('Contrast scaling parameters: %f, %f, %f, %f'%(params[0],params[1],params[2],params[3],))
                    input = self.apply_region_wise_contrast_modulation(input, region_map, params, self.blur_radius_for_region_map)
                    # if debugging: print('After contrast scaling, min: %f max: %f'%(input.min(),input.max()))
                    if drawSample:
                        img = Image.fromarray(np.squeeze(input[0].data.cpu().numpy()))
                        img = img.convert('RGB')
                        img.save('preprocess2_contrast_scaled_img.png')
                if debugging: print('contrast scaling takes: %.5f'%(time.time() - block_start_time))

                block_start_time = time.time()
                if 'blur_radii' in self.config.augmentations:
                    params = np.asarray(self.config.augmentations['blur_radii'])
                    if 'blur_radii_perturbations' in self.config.augmentations:
                        params = params + np.asarray(self.config.augmentations['blur_radii_perturbations']) * (2 * np.random.random(region_num) -1)
                    # if debugging: print('Blur radii: %f, %f, %f, %f'%(params[0],params[1],params[2],params[3],))
                    input = self.apply_region_wise_blur(input, region_map, params, self.blur_radius_for_region_map)
                    # if debugging: print('After blurring, min: %f max: %f'%(input.min(),input.max()))
                    if drawSample:
                        img = Image.fromarray(np.squeeze(input[0].data.cpu().numpy()))
                        img = img.convert('RGB')
                        img.save('preprocess3_blur_applied_img.png')
                if debugging: print('blurring takes: %.5f'%(time.time() - block_start_time))

            block_start_time = time.time()
            if do_augmentation and 'global_gaussian_noise' in self.config.augmentations:
                # if debugging: print('Global Gaussian noise level: %f'%self.config['global_gaussian_noise'])
                input = self.apply_global_gaussian_noise(input, self.config.augmentations['global_gaussian_noise'])
                # if debugging: print('After adding Gaussian noise, min: %f max: %f'%(input.min(),input.max()))
                if drawSample:
                    img = Image.fromarray(np.squeeze(input[0].data.cpu().numpy()))
                    img = img.convert('RGB')
                    img.save('preprocess4_gaussian_noise_applied_img.png')
            if debugging: print('Gaussian noising takes: %.5f'%(time.time() - block_start_time))

            block_start_time = time.time()
            if self.config.augmentations['normalize_input']:
                global_contrast_perturbation = 0.0
                if ('global_contrast_perturbation' in self.config.augmentations) and do_augmentation:
                    global_contrast_perturbation = self.config.augmentations['global_contrast_perturbation']
                # if debugging: print('Global contrast perturbation magnitude: %f'%global_contrast_perturbation)
                if self.config.network_type.__name__ == 'ResNet50' or self.config.network_type.__name__ == 'ResNet50FeaturesFrozen': # run preprocessing that's specific to ResNet50
                    input = self.preprocessing_ResNet50(input)
                else:
                    input = self.normalize_with_contrast_modulation(input, global_contrast_perturbation)
                if drawSample:
                    img = Image.fromarray(np.squeeze((1 + input[0].data.cpu().numpy()) * 255 / 2))
                    img = img.convert('RGB')
                    img.save('preprocess5_global_contrast_normalization_applied_img.png')
            if debugging: print('Global normalization takes: %.5f'%(time.time() - block_start_time))

            # Aperture masking should be done regardless of the do_augmentation flag.
            if self.config.network_type.__name__ == 'ResNet50' or self.config.network_type.__name__ == 'ResNet50FeaturesFrozen': # run preprocessing that's specific to ResNet50
                outputs.data[i_from:i_to] = (input * self.aperture_mask).data[0:(i_to - i_from)] # fill 0 outside the valid aperture area
            else:
                outputs.data[i_from:i_to] = (input * self.aperture_mask).data[0:(i_to - i_from)] # fill 0 outside the valid aperture area
            # inputs[ii] = input + (Variable(torch.ones(input.shape)).cuda() - self.aperture_mask) * 0.5

            # below was for debugging
            # if self.num_sample_store > 0 and do_augmentation:
            #     for ii in range(outputs.shape[0]):
            #         img = Image.fromarray(np.squeeze((outputs.data[ii,:].cpu().numpy() + 1) * 255 / 2).astype(np.uint8))
            #         # img.save('augmented_sample_%d.png'%self.num_sample_store)
            #         self.num_sample_store -= 1
            #         if self.num_sample_store == 0:
            #             break

        return outputs
