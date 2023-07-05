# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from itertools import zip_longest as itzl
import logging
import pdb
import torch
import random
import math
import cv2 as cv
from PIL import Image
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
from utils import resizeImage, resizeImgChannel
from torchvision import transforms, utils

datasetRootPath = 'C:/NvGaze/Datasets/h5/'

class Dataset():
    """docstring for GazeNetDataLoader"""
    def __init__(self, data_path, outputs = ['directions'], batch_size = 50, input_resolution = (255,191), augmentation = True, usedDataPercentage = 100.0):
        self.batch_size = batch_size
        assert type(data_path).__name__ == 'str', "data_path must be a string"

        # Initialize all attributes to None
        self.aperture_masks = None
        self.region_maps = None
        self.all_info = {}

        self.performAugmentation = augmentation

        images = None # temporary data

        # For safety, use context management idiom "with"
        with h5py.File(data_path,'r') as f:

            print('Loading h5 file: ', os.path.basename(data_path))
            print('Required image resolution: ', input_resolution)
                        
            # generate indices for samples
            datasetlength = f['images'].shape[0]
            numUsedSamples = int(datasetlength * (usedDataPercentage / 100.0))
            allsamples = np.arange(datasetlength)
            np.random.shuffle(allsamples) # shuffling is performed in-place
            selectedSamples = allsamples[0:numUsedSamples]
            selectedSamples = np.sort(selectedSamples)   
            
            # load images (read into system memory)
            images = f['images'][selectedSamples,...]

            h5_resolution = (images.shape[-1],images.shape[-2])

            print('selected samples')
            print(selectedSamples)
            print('Image resolution in h5 file: ', h5_resolution)
            print("images loaded with shape : ", images.shape)
            print("images datatype : ", images.dtype)

            # resize images and region maps if required
            if input_resolution[0] != h5_resolution[0] or input_resolution[1] != h5_resolution[1]: 

                logging.debug('   Resizing images to the specified resolution...')
                print('   h5 image resolution : ', images.shape)

                new_resolution_dims = list(images.shape);
                new_resolution_dims[-2] = input_resolution[1] # height
                new_resolution_dims[-1] = input_resolution[0] # width

                print('   new image stack shape : ', new_resolution_dims)
                rescaled_images = np.zeros(tuple(new_resolution_dims), dtype=images.dtype)

                print('   Resizing images... ')
                for image_idx in range(images.shape[0]):
                    rescaled_images[image_idx] = resizeImage(images[image_idx], (input_resolution[0], input_resolution[1]))
                logging.debug('   Resizing complete. Copying resized images.')
                images = rescaled_images

            # validate format of loaded images
            print("loaded images shape  in NEVCHW: ", images.shape)
            assert(len(images.shape) == 6), "images have to be loaded in NEVCHW"

            # load aperture masks
            if 'aperture_masks' in f:
                masks = f['aperture_masks'][selectedSamples,...]
                self.aperture_masks = np.asarray(masks, dtype=np.float32)
                self.aperture_masks = self.aperture_masks / 255.0 # normalize
            else:
                print ("WARNING : no aperture masks contained in dataset !")
                # TODO initialize default aperture masks in format EVCHW
                self.aperture_masks = np.ones(images[0].shape, np.float32) # already normalized
                print('using default aperture masks of shape : ', self.aperture_masks.shape)
            
            # resize aperture masks if required
            if self.aperture_masks.shape[-1] != images.shape[-1] or self.aperture_masks.shape[-2] != images.shape[-2]: 
                print("resizing aperture masks")
                resizedApertureMasks = resizeImage(self.aperture_masks, (input_resolution[0], input_resolution[1]), 'nearest')
                self.aperture_masks = resizedApertureMasks

            print("aperture masks shape in EVCHW: ", self.aperture_masks.shape)
            assert(len(self.aperture_masks.shape) == 5), "aperture masks have to be loaded in EVCHW"
                        
            # all_info is a dictionary containing all the descriptive values of input data (i.e. content in the csv file in footage).
            # This can be useful when we need information other than labels (e.g. multi-subject network needs subject ID to choose the corresponding calibration network).
            for key in f.keys():
                
                if (key not in outputs):
                    print('skipping the following key which is not a required label : ' + key)
                    continue

                print('loading label : ' + key)

                np_values = np.asarray(f[key][selectedSamples,...])

                if key in self.all_info:
                    if len(np_values.shape) == 1:
                        self.all_info[key] = np.concatenate((self.all_info[key], np.expand_dims(np_values, axis = 1)), axis = 0)
                    else:
                        self.all_info[key] = np.concatenate((self.all_info[key], np_values), axis = 0)
                else: # add key with values
                    self.all_info[key] = np_values

                if len(self.all_info[key].shape) == 1:
                    self.all_info[key] = np.expand_dims(self.all_info[key], axis = 1)

            print ("Supported Keys: ", [x for x in self.all_info.keys()])

            # create labels depending on what 'outputs' is.
            if outputs == None: # by default label is gaze direction
                if 'directions' in [str(x) for x in f.keys()]: # if this is an h5 dataset with new format
                    self.labels = self.all_info['directions']
                #else: # old format
                    self.labels = self.all_info
                #
                self.labels = np.asarray(self.labels) # convert to numpy array
            else: # when custom label is given.

                self.labels = []
                ## check if all required labels are contained
                for o in outputs:
                    assert(o in self.all_info), ("Required label '%s' could not be found in provided labels of h5 dataset file" % (o))
                    if len(self.labels) == 0:
                        self.labels = self.all_info[o]
                    else:
                        self.labels = np.concatenate((self.labels, self.all_info[o]), axis = 1)


        # all indices -> used for iteration over dataset
        self.indices = np.arange(images.shape[0])

        # load all images in NCHW (already in system memory but size may be a problem for large datasets !)
        self.images_np = np.asarray(images[:,0,0,...]) # NCHW

        # image resolution of (potentially resized) input images
        self.resX = int(self.images_np.shape[-1])
        self.resY = int(self.images_np.shape[-2])
                
        # load all labels in system memory
        self.labels_np = self.labels.astype(np.float32)
        print("labels shape : ", self.labels_np.shape)

        self.numSamplesInDataset = images.shape[0]

        ## DATA AUGMENTATION

        self.cropResolution = (293,293) # region size for cropped pupil region
        self.cropResolutionHalf = (float(self.cropResolution[0]) / 2.0, float(self.cropResolution[1] / 2.0))
        
        self.validPupilRangeX = (0.1,0.9)#(0.25,0.75) # only accept sample if original (non-transformed) pupil is in this range
        self.validPupilRangeY = (0.1,0.9)#(0.3,0.70) # only accept sample if original (non-transformed) pupil is in this range

        self.padwidth = math.ceil(np.max(self.cropResolution)) # how much padding has to be added ?

        self.validPupilRangeX = (self.validPupilRangeX[0]*self.resX, self.validPupilRangeX[1]*self.resX)
        self.validPupilRangeY = (self.validPupilRangeY[0]*self.resY, self.validPupilRangeY[1]*self.resY)

        self.useHistogramEqualization           = False;

        if (self.performAugmentation == True):

            # parameters for augmentations

            self.useAffineTransformRandomization    = True
            self.maxTranslation                     = 80    # lower number for pupil center shift
            self.maxAngle                           = 45    # maybe higher angle here

            self.minScale                           = 1.0
            self.maxScale                           = 1.0

            self.usePupilPositionLimit              = True

            self.useIntensityNoise                  = True
            self.intensity_max_noise                = 10    # intensity noise

            self.useGlobalOffsetRandomization       = True
            self.maxIntensityOffset                 = 40

            self.useGaussianNoise                   = True
            self.max_sigma                          = 1.0   # random gaussian
            
            self.useRandomizedShrinking             = True
            self.shrink_limit                       = 0.25  # range of valid downscaling

            self.useRandomizedReflections            = True
            self.probabilityReflections             = 0.25
            self.opacityRange                       = (0.1, 0.2) # (0.1, 0.5)
            self.useRandReflectionJitter            = False
            self.maxBrigthnessOffset                = 20
            self.maxContrastScale                   = 2.0
            self.useReflectionClipping              = True
            
            self.useNormalizationRandomization      = True
            self.lowerPertubation                   = 0.1
            self.meanPertubation                    = 0.15
            self.upperPertubation                   = 0.1

            # load reflection images
            self.reflectionImages = None

            global datasetRootPath

            if (self.useRandomizedReflections):
                print('loading reflection images from h5 file...')
                self.reflectionImages = []
                reflectionsFilePath = os.path.join(datasetRootPath, "reflectiondataset.h5")
                with h5py.File(reflectionsFilePath, 'r') as h5f:
                    for key in h5f.keys():
                        imgPtr = h5f[key]
                        self.reflectionImages.append(imgPtr[:])
                print('%d reflection images loaded...'%(len(self.reflectionImages)))


        print('dataset loaded...')

    def interpolateLinear(self, T_norm, lowerLimit, upperLimit, mask = None):

        if (mask is not None):
            return (lowerLimit * (-1.0 * T_norm + 1.0) + T_norm * upperLimit) * mask
        # without mask
        return lowerLimit * (-1.0 * T_norm + 1.0) + T_norm * upperLimit

    def advanced_normalization(self, inputs, lowerPertub, meanPertub, upperPertub):

        # mem copy tensor
        outputs = inputs

        # iterate over input images by index (N)
        for inputImageIndex in range(inputs.shape[0]): 
            
            # iterate over each channel (C) -> assuming one grayscale image and image channel per channel
            for channelIndex in range(inputs.shape[1]): 

                # get image
                img_normalized = inputs[inputImageIndex][channelIndex] / 255.0 # HW
                           
                # randomize pertubation
                lower_pertubation = random.uniform(-lowerPertub, lowerPertub)   # randomize lower intensity limit
                center_pertubation = random.uniform(-meanPertub, meanPertub)  # randomize center intensity
                upper_pertubation = random.uniform(-upperPertub, upperPertub)   # randomize upper intensity limit
                    
                # compute mask based on value interval (encoded by sign)
                img_normalized_sign = torch.sign(img_normalized - 0.5)
                img_mask_interval0 = torch.clamp(img_normalized_sign * -1.0, min = 0, max = 1.0) # 1 if value is in interval [0,0.5]
                img_mask_interval1 = torch.clamp(img_normalized_sign, min = 0, max = 1.0)        # 1 if value is in interval [0.5,1.0]
                                                        
                # apply pertubation in normalized space

                # lower pertubation -> for values in interval [0, 0.5]
                lowerPertub = self.interpolateLinear(img_normalized * 2,
                                                2.0 * lower_pertubation,
                                                2.0 * center_pertubation + 1.0,
                                                img_mask_interval0)
                lowerPertub = lowerPertub / 2.0 # rescale to [0,0.5]

                # upper pertubation -> for values in interval [0, 0.5]
                upperPertub = self.interpolateLinear((img_normalized - 0.5) * 2,
                                                2.0 * center_pertubation,
                                                2.0 * upper_pertubation + 1.0,
                                                img_mask_interval1)

                upperPertub = ((upperPertub / 2.0) + 0.5) * img_mask_interval1 # rescale to [0.5,1.0]

                # result is just sum of both since each result has been masked before (works because values are masked out)
                # shift range from [0,1] to [-1,1]
                img_normalized_masked = ((lowerPertub + upperPertub) * 2.0) - 1.0 

                outputs[inputImageIndex][channelIndex] = img_normalized_masked # write result
       
        return outputs

    # Shuffle the images within the input array, discards clip information
    def shuffle(self):
        np.random.shuffle(self.indices) # shuffling is performed in-place

    def __len__(self): # return the number of individual data
        return int(np.ceil(self.numSamplesInDataset / self.batch_size))

    def __getitem__(self,idx):

        # blending reflection
        def blendReflection(base, blend, opacity):
            #Scale values to 0-1
            base = np.asarray(base, dtype=np.float32) / 255.0
            blend = np.asarray(blend, dtype=np.float32) / 255.0
            
            oneArray = np.full(base.shape, 1.0, dtype=np.float32)
            baseAndBlend = np.multiply(np.subtract(oneArray, base), np.subtract(oneArray, blend))
            blendImage = np.subtract(oneArray, baseAndBlend)

            #And scale back up
            blended_array = np.asarray(np.add((blendImage * opacity), base * (1.0 - opacity)) * 255, dtype=np.uint8)
            #pdb.set_trace()
            return blended_array
       
        # fetch indices
        i_from = int(self.batch_size * idx) # create start index from current batch index and batch size
        i_to = int(np.min([self.numSamplesInDataset,i_from + self.batch_size])) # create end index (with handling length of dataset)
        
        shuffledIndices = self.indices[i_from:i_to] # get start/end indices from shuffled indices
        
        # save only valid images+labels
        validSamples = list(range(self.batch_size))
        validImages = []
        validLabels = []

        # get images        
        images_tensor = np.copy(self.images_np[shuffledIndices,...]) # work on copy
        # get labels
        labels_batch = self.labels_np[shuffledIndices,...]
       
        labelsNormalzed = False

        # augmentations for training images
        if (self.performAugmentation):
            
            for i in range(len(shuffledIndices)):

                # get labels -> usually pupil location in full image space
                pupilx_original = labels_batch[i,0]
                pupily_original = labels_batch[i,1]

                # check for valid range
                if (pupilx_original >= self.validPupilRangeX[0] and pupilx_original < self.validPupilRangeX[1] and pupily_original >= self.validPupilRangeY[0] and pupily_original < self.validPupilRangeY[1]):
                    validSamples[i] = True
                else:
                    validSamples[i] = False
                    continue

                # get 2d image as numpy array
                img = images_tensor[i,0] # uint8 image

                dx = 0 # applied shift to the pupil location in x
                dy = 0 # applied shift to the pupil location in y

                if (self.useAffineTransformRandomization):

                    transformation_Matrix = None

                    while(True):

                        # randomize affine transform parameters
                        randAngle = math.radians(random.uniform(-self.maxAngle,self.maxAngle))
                        randTranslateX = random.uniform(-self.maxTranslation,self.maxTranslation)
                        randTranslateY = random.uniform(-self.maxTranslation,self.maxTranslation)
                        randScaleFactor = random.uniform(self.minScale,self.maxScale)

                        dx = randTranslateX
                        dy = randTranslateY

                        # set up affine transform with normalized coordinates                    
                        cos = math.cos(randAngle) * randScaleFactor
                        sin = math.sin(randAngle)

                        # transformation of labels has to happen different depending if the pupil location is normalized or not
                        transformation_Matrix = None
                        minPPositionX = None
                        maxPPositionX = None
                        minPPositionY = None
                        maxPPositionY = None
                        
                        # rotation about center
                        M1 = np.matrix([[1, 0, -pupilx_original],[0, 1, -pupily_original],[0,0,1]],dtype=np.float)
                        M2 = np.matrix([[cos, sin, 0],[-sin, cos, 0],[0,0,1]],dtype=np.float)
                        S = np.matrix([[randScaleFactor, 0, 0],[0, randScaleFactor, 0],[0,0,1]],dtype=np.float)
                        M3 = np.matrix([[1, 0, randScaleFactor*pupilx_original],[0, 1, randScaleFactor*pupily_original],[0,0,1]],dtype=np.float)
                        M4 = np.matrix([[1, 0, randScaleFactor*dx],[0, 1, randScaleFactor*dy],[0,0,1]],dtype=np.float)

                        transformation_Matrix = np.matmul(np.matmul(np.matmul(np.matmul(M4,M3),S),M2),M1)

                        validPupilRangeInCroppedAreaNormalized = 0.75

                        minPPositionX = pupilx_original - (self.cropResolutionHalf[0] * validPupilRangeInCroppedAreaNormalized)
                        maxPPositionX = pupilx_original + (self.cropResolutionHalf[0] * validPupilRangeInCroppedAreaNormalized)
                        minPPositionY = pupily_original - (self.cropResolutionHalf[1] * validPupilRangeInCroppedAreaNormalized)
                        maxPPositionY = pupily_original + (self.cropResolutionHalf[1] * validPupilRangeInCroppedAreaNormalized)

                        labelmatrix = np.matmul(np.matrix([pupilx_original, pupily_original, 1.0]), np.transpose(transformation_Matrix))[0:2] # multiply label with affine transform 
                        transformed_labels = (labelmatrix[0,0], labelmatrix[0,1])

                        dx = pupilx_original - transformed_labels[0]
                        dy = pupily_original - transformed_labels[1]

                        if (self.usePupilPositionLimit):
                            if (minPPositionX < transformed_labels[0] < maxPPositionX and minPPositionY < transformed_labels[1] < maxPPositionY):
                                break # quit while loop
                        else:
                            break # quit while loop
                        
                    # apply affine transform
                    cv.warpAffine(src=img,M=transformation_Matrix[0:2,0:3],dsize=(self.resX,self.resY),dst=img,flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT)
                 

                # //////////////////////////
                # crop to pupil region
                # //////////////////////////

                # get crop limits
                leftX = int(pupilx_original - self.cropResolutionHalf[0])
                rightX = leftX + int(self.cropResolution[0])
                topY = int(pupily_original - self.cropResolutionHalf[1])
                bottomY = topY + int(self.cropResolution[1])
                cropROI = [leftX,rightX,topY,bottomY] # RECT

                # get images with padding
                np_with_pad = np.pad(img, pad_width=self.padwidth, mode='mean') # reflect

                # get cropped pupil region
                img = np_with_pad[cropROI[2]+self.padwidth:cropROI[3]+self.padwidth, cropROI[0]+self.padwidth:cropROI[1]+self.padwidth]

                validLabels.append(np.asarray([self.cropResolutionHalf[0] - dx,self.cropResolutionHalf[1] - dy], dtype=np.float32))
                # //////////////////////////

                if (self.useIntensityNoise or self.useGlobalOffsetRandomization):

                    # add global offset and intensity noise
                    globalOffset = False
                    randomIntensityNoise = False
                    if (random.uniform(0,1.0) > 0.5):
                        globalOffset = True
                    if (random.uniform(0,1.0) > 0.5):
                        randomIntensityNoise = True

                    # add intensity noise
                    if (randomIntensityNoise or globalOffset):
                        image_int16 = img.astype(np.int16)
                        if (randomIntensityNoise):
                            image_int16 +=  np.random.randint(low=-self.intensity_max_noise, high=self.intensity_max_noise, size=img.shape, dtype=np.int16)
                        if (globalOffset):
                            image_int16 += math.floor(random.uniform(-self.maxIntensityOffset,self.maxIntensityOffset))
                        cv.normalize(image_int16, image_int16, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
                        img = image_int16.astype(np.uint8)

                # add gaussian noise
                if (self.useGaussianNoise):
                    if (random.uniform(0,1.0) > 0.5):
                        sigma = np.random.uniform(low=0.1, high=self.max_sigma) 
                        img = cv.GaussianBlur(img, (7, 7), 0.5 + np.abs(sigma))
                            
                # randomly scale small
                if (self.useRandomizedShrinking):
                    if (random.uniform(0,1.0) > 0.5):
                        rescale_noise = np.random.uniform(low=self.shrink_limit, high=1.0)
                        img = cv.resize(img, dsize=(0, 0), fx=rescale_noise, fy=rescale_noise,interpolation=cv.INTER_CUBIC)
                        img = cv.resize(img, dsize=(self.cropResolution[0], self.cropResolution[1]), interpolation=cv.INTER_CUBIC)
                            
                # add reflections
                if (self.useRandomizedReflections):
                    if (random.uniform(0,1.0) < self.probabilityReflections):
                            
                        reflectionImage = random.choice(self.reflectionImages)

                        # random crop reflection image
                        x = random.randint(0, reflectionImage.shape[1] - self.cropResolution[0])
                        y = random.randint(0, reflectionImage.shape[0] - self.cropResolution[1])

                        reflectionImage = reflectionImage[y:y+self.cropResolution[1], x:x+self.cropResolution[0]]

                        # color jitter
                        if (self.useRandReflectionJitter):
                            intensityShift = max(0,min(255,math.floor(random.uniform(-self.maxBrigthnessOffset,self.maxBrigthnessOffset))))
                            contrastScale = max(0,min(255,math.floor(random.uniform(1.0, self.maxContrastScale))))
                            reflectionImage = cv.convertScaleAbs(reflectionImage, -1, contrastScale, intensityShift);

                        if (self.useReflectionClipping):
                            np.clip(reflectionImage, 0, 255, out=reflectionImage) # in-place, force all values to be between 0 and 255

                        opacity = random.uniform(self.opacityRange[0], self.opacityRange[1])

                        img = blendReflection(img, reflectionImage, opacity)        
                        

                # histogram equalization
                if (self.useHistogramEqualization):
                    img = cv.equalizeHist(img)

                # write images and labels
                validImages.append(np.expand_dims(img,0))

                        
        else:

            # augmentations for test images
            for i in range(len(shuffledIndices)):

                # get labels -> usually pupil location in full image space
                pupilx = labels_batch[i,0]
                pupily = labels_batch[i,1]

                # check for valid range
                if (pupilx >= self.validPupilRangeX[0] and pupilx < self.validPupilRangeX[1] and pupily >= self.validPupilRangeY[0] and pupily < self.validPupilRangeY[1]):
                    validSamples[i] = True
                else:
                    validSamples[i] = False
                    continue

                dx = 0
                dy = 0

                # get crop limits
                leftX = int(labels_batch[i,0] - self.cropResolutionHalf[0])
                rightX = leftX + int(self.cropResolution[0])
                topY = int(labels_batch[i,1] - self.cropResolutionHalf[1])
                bottomY = topY + int(self.cropResolution[1])
                cropROI = [leftX,rightX,topY,bottomY] # RECT

                # get images with padding
                np_with_pad = np.pad(images_tensor[i,0,...], pad_width=self.padwidth, mode='mean') # reflect

                # get cropped pupil region
                img = np_with_pad[cropROI[2]+self.padwidth:cropROI[3]+self.padwidth, cropROI[0]+self.padwidth:cropROI[1]+self.padwidth]

                # histogram equalization
                if (self.useHistogramEqualization):
                    img = cv.equalizeHist(img)

                # write images and labels
                validImages.append(np.expand_dims(img,0))
                validLabels.append(np.asarray([self.cropResolutionHalf[0] + dx,self.cropResolutionHalf[1] + dy], dtype=np.float32))
                

        region_maps = None # return empty list if the dataset does not contain region_maps
       

        # return None if there were no valid images
        if (len(validImages) == 0):
            return None, None, None

        # upload images to GPU
        images_gpu = torch.from_numpy(np.asarray(validImages,dtype=np.float32)).cuda()
        labels_gpu = torch.from_numpy(np.asarray(validLabels,dtype=np.float32)).cuda()

        # normalize (on GPU)
        if (self.performAugmentation and self.useNormalizationRandomization):
            images_gpu = self.advanced_normalization(images_gpu, self.lowerPertubation, self.meanPertubation, self.upperPertubation)
        else: # always apply normalization
            images_gpu = (2.0 * (images_gpu / 255.0)) - 1.0

        return images_gpu, region_maps, labels_gpu
