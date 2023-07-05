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

import os, sys, argparse, platform, csv, time, re
import numpy as np
import imageio as iio
import glob
import h5py
import codecs
import pdb
import time
import math
import cv2
import itertools
import psutil
from matplotlib import pyplot as plt
import zipfile
from io import BytesIO
import datetime

# import utils
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
from utils import generateSampleIndices, getH5Datatype, resizeImage, importCSVHeader, parseCSV

import re

#----------------------------------------------------------------------------
def generateRegionMap(region_mask, skinless_region_mask, mask_img):
    
    # normalize aperture mask
    aperture_mask = None
    if (mask_img is not None):
        aperture_mask = np.round(np.array(mask_img) / 255.0).astype(bool)
    else:
        aperture_mask = np.ones((region_mask.shape[1],region_mask.shape[2]), dtype=np.uint8).astype(bool)

    #print("aperture mask shape : ", mask_img.shape)

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
    rm0 = region_mask[0,:,:]
    rm1 = region_mask[1,:,:]
    rm2 = region_mask[2,:,:]
    sm0 = skinless_region_mask[0,:,:]
    sm1 = skinless_region_mask[1,:,:]
    sm2 = skinless_region_mask[2,:,:]

    sclera_mask = np.logical_and(sm0>0, aperture_mask)
    iris_mask = np.logical_and(sm1 > 250, aperture_mask)
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

    regionMap = np.asarray(pupil_map + iris_map + sclera_map + skin_map + glint1_map + glint2_map + glint3_map + glint4_map + glint5_map, dtype=np.int16)

    return regionMap

def loadApertureMasks(footagepath, use_masks = True):
    #  get mask images
    mask_imgs = None
    left_eye_mask_imgs = None
    right_eye_mask_imgs = None

    if use_masks:
        mask_pair = None
        # Attempt to read mask image from footage folder
        mask_files = glob.glob(os.path.join(footagepath, '*mask*.*'))
        print('Using mask images:', mask_files)

        

        # check images for left eye
        for file in mask_files:
            print(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            parts = filename.split("_")
            print(parts)

            if (parts[0] == "mask"):
                if (parts[1] == "L"):
                    print("L view : ", parts[2])
                    print("loading mask : ", file)
                    mask = np.asarray(iio.imread(file), dtype=np.uint8)

                    # make sure file is in CHW
                    if (len(mask.shape) == 2):
                        mask = np.expand_dims(mask, 0)
                        print("expanding image dimension")
                    assert (len(mask.shape) == 3), "ERROR: unknown format if aperture mask image !"

                    if (mask.shape[0] > 3):                     # deal with HWC to CHW conversion
                        mask = np.transpose(mask, [2,0,1])

                    mask = np.expand_dims(mask, 0)      # add dimension for view
                    print("mask shape View-Channel-Height-Width : ", mask.shape)
                    # add mask to list of masks for left eye
                    if (left_eye_mask_imgs is None):
                        left_eye_mask_imgs = mask
                    else:
                        left_eye_mask_imgs = np.concatenate((left_eye_mask_imgs, mask), 0)
     
        if (left_eye_mask_imgs is not None):
            print("mask shape after checking for left eye : View-Channel-Height-Width : ", left_eye_mask_imgs.shape)
        else:
            print("no aperture masks for left eye found !")

        # check images for right eye
        for file in mask_files:
            print(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            parts = filename.split("_")
            print(parts)

            if (parts[0] == "mask"):
                if (parts[1] == "R"):
                    print("R view : ", parts[2])
                    print("loading mask : ", file)
                    mask = np.asarray(iio.imread(file), dtype=np.uint8)

                    # make sure file is in CHW
                    if (len(mask.shape) == 2):
                        mask = np.expand_dims(mask, 0)
                        print("expanding image dimension")
                    assert (len(mask.shape) == 3), "ERROR: unknown format of aperture mask image !"

                    if (mask.shape[0] > 3):                     # deal with HWC to CHW conversion
                        mask = np.transpose(mask, [2,0,1])

                    mask = np.expand_dims(mask, 0)      # add dimension for view
                    print("mask shape View-Channel-Height-Width : ", mask.shape)
                    # add mask to list of masks for left eye
                    if (right_eye_mask_imgs is None):
                        right_eye_mask_imgs = mask
                    else:
                        right_eye_mask_imgs = np.concatenate((right_eye_mask_imgs, mask), 0)
        
        if (right_eye_mask_imgs is not None):
            print("mask shape after checking for right eye : View-Channel-Height-Width : ", right_eye_mask_imgs.shape)
        else:
            print("no aperture masks for right eye found !")

    # stack left and right view images into 'Eye View Channel Height Width' format
    if (left_eye_mask_imgs is not None):
        mask_imgs = np.expand_dims(left_eye_mask_imgs, 0)

    if (right_eye_mask_imgs is not None):
        if (mask_imgs is None):
            mask_imgs = np.expand_dims(right_eye_mask_imgs, 0)        
        else:
            mask_imgs = np.concatenate((mask_imgs, np.expand_dims(right_eye_mask_imgs, 0)), 0)

    if (mask_imgs is not None):
        print("mask shape of left/right masks : Eye-View-Channel-Height-Width : ", mask_imgs.shape)

    return mask_imgs

def maskAndResize(img, mask_imgs, eye, output_resolution):
    img = resizeImage(img, output_resolution)
    # Check if we should apply an image mask.
    # img is in WH(maybe C) format
    if mask_imgs is not None:

        # If mask image has multiple channels, only use the first channel
        if len(mask_imgs[0].shape) == 3:
            mask_imgs = (mask_imgs[0][0, ...], mask_imgs[1][0, ...])

        # If input image is multi-channel, apply the mask to all channels
        mask = None
        if eye == "L":
            mask = mask_imgs[0]
        elif eye == "R":
            mask = mask_imgs[1]
        else:
            assert False, "Eye label must be 'L' or 'R'"
            
        multichannelmask = np.repeat([mask], img.shape[0], axis=0)
        #multichannelmask = np.transpose(multichannelmask, (1, 2, 0))
        

        # apply mask using float precision and convert back to uint8 
        img = (img.astype(np.float32) * (multichannelmask.astype(np.float32) / 255.0)).astype(np.uint8)

    # resize image and return

    return img

def generateSemanticLabels(regionmask):
    # get semantic-specific masks. regionmask is assumed to be a 3D tensor (CHW)

    # Math is simpler if regionmask is 2D for semantic label calculation.
    # Also, opencv function 'FindContours' requires 2D matrix.
    # Hence generate a 2D region mask that is a local variable
    regionmask2D = regionmask[0]

    # define x and y coordinate of the image. We follow the mathematical convention for pixel address (bottom left corner being the origin)
    xv, yv = np.meshgrid(np.arange(0,regionmask2D.shape[1],1),np.arange(regionmask2D.shape[0]-1,-1,-1))

    # extract unoccluded pupil are and open pupil area
    unoccluded_pupil_mask = np.mod(regionmask2D,10) == 1
    open_pupil_mask = np.mod(regionmask2D,100) == 1

    # get pupil occclusion ratio and pupil center position using the pupil masks
    pupil_occlusion_ratio = np.sum(open_pupil_mask) / np.sum(unoccluded_pupil_mask)
    pupil_center_x = np.sum(xv * unoccluded_pupil_mask)/np.sum(unoccluded_pupil_mask)
    pupil_center_y = np.sum(yv * unoccluded_pupil_mask)/np.sum(unoccluded_pupil_mask)

    # fit an ellipse to the pupil using opencv functions. First, get pupil image
    pupil_img = unoccluded_pupil_mask.astype(np.uint8) * 255
    _, contours, _ = cv2.findContours(pupil_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # choose the contour with the most number of elements.
    c_idx = 0
    c_size = 0
    for contour_idx, contour in enumerate(contours):
        if contour.size > c_size:
            c_idx = contour_idx
            c_size = contour.size

    try:
        pupil_pos, pupil_axis, pupil_angle = cv2.fitEllipse(contours[c_idx])
    except:
        pdb.set_trace() # this should not happen...

    return pupil_occlusion_ratio, pupil_center_x, pupil_center_y, pupil_pos[0], pupil_pos[1], pupil_axis[0], pupil_axis[1], pupil_angle

def writeH5File(h5filepath, labels, labellist, sampleIndices, footagepath, output_resolution, mask_imgs, eyes, use_region_masks, convertToGrayscale ):

    print("write h5 file : ", h5filepath)

    # write h5 file
    with h5py.File(h5filepath, 'w') as h5_file:

        # write meta data
        metadata = {'eyes': eyes, 'image_resolution': output_resolution }
        h5_file.attrs.update(metadata)

        # get image format in N E V C H W format
        
        eye_view_labels = list()
        eye_view_labels_left = list()
        eye_view_labels_right = list()
        numViewsLeft = 0
        numViewsRight = 0

        # analyse available labels to estimate number of eyes and views
        for label in labellist:
            labelprefix = label.split("_") # split string at given character

            if (len(labelprefix) > 2): # i.e. image_L_0 (left image, first view)
                if (labelprefix[-2] == "L"):
                    eye_view_labels.append(label)
                    eye_view_labels_left.append(label)
                    #print(labelprefix[0])
                    if (labelprefix[0] == "image"):
                        numViewsLeft += 1

                if (labelprefix[-2] == "R"):
                    eye_view_labels.append(label)
                    eye_view_labels_right.append(label)
                    #print(labelprefix[0])
                    if (labelprefix[0] == "image"):
                        numViewsRight += 1
            #else:
            #    print("other label : ", labelprefix)

        # get eyes
        print("left eye labels : ", eye_view_labels_left)
        print("right eye labels : ", eye_view_labels_right)

        # get number of views
        print("numViewsLeft : ", str(numViewsLeft))
        print("numViewsRight : ", str(numViewsRight))
                
        # assert that number of views per eye match !
        if (numViewsLeft > 0 and numViewsRight > 0):
            if (numViewsLeft != numViewsRight):
                printf("ERROR: images for left and right views given but number of views does not match !") # print error message also when assert message is not printed
                assert (numViewsLeft == numViewsRight), "ERROR: images for left and right views given but number of views does not match !"
            if (len(eye_view_labels_left) != len(eye_view_labels_right)):
                printf("ERROR: images for left and right views given but available labels do not match !") # print error message also when assert message is not printed
                assert (len(eye_view_labels_left) == len(eye_view_labels_right)), "ERROR: images for left and right views given but available labels do not match !"


        # N : number of samples
        N = len(sampleIndices)
        # E : available eyes
        E = 0
        # V : available views per eye
        V = 0

        if (numViewsLeft > 0):
            E += 1
            V = numViewsLeft
        if (numViewsRight > 0):
            E += 1
            V = numViewsRight # assumes no or equal number of views for left eye

        # C : channels per image
        C = 0
        # H : image height
        H = 0
        # W : image width
        W = 0
        
        # open test image to receive channel/height/width format
        openedArchive = None # initialize zip archive path
        zip = None # initialize zip archive to be opened
        archivename = str(labels['archive'][sampleIndices[0]])  # get archive path of current image 
        
        #archivepath = np.array2string(archivename)[1:-1]
        archivepath = archivename
        if (os.path.exists(archivepath) == False):
            archivepath = os.path.join(footagepath, archivepath)

        # open required archive if it is not yet opened
        if (openedArchive != archivepath):
            # close archive if opened
            if (zip is not None):
                zip.close()
            # open archive
            zip = zipfile.ZipFile(archivepath, 'r')
            openedArchive = archivepath
            if (numViewsLeft > 0):
                imagename = np.array2string(labels[eye_view_labels_left[0]][sampleIndices[0]])
            elif (numViewsRight > 0):
                #print(eye_view_labels_right[0])
                imagename = np.array2string(labels[eye_view_labels_left[1]][sampleIndices[0]])
                #print(imagename[1:-1])
            img = iio.imread(BytesIO(zip.open(imagename[1:-1], 'r').read()))

            if (len(img.shape) == 3):
                if (img.shape[0] > 4):
                    print("tranpose required for image file : ", imagename)
                    img = np.transpose(img, [2, 0, 1])
                C = img.shape[0]
                H = img.shape[1]
                W = img.shape[2]
                print("3 image dimensions : ", img.shape)
            elif (len(img.shape) == 2):
                C = 1
                H = img.shape[0]
                W = img.shape[1]
                print("2 image dimensions : ", img.shape)
                print("assuming C = 1")
            else:
                print("ERROR: input images have to be 3-dimensional or 2-dimensional but given number of dimension = ", img.shape)
                assert(False), "ERROR: input images have to be 3-dimensional or 2-dimensional but given number of dimension = %d" %(img.shape)

            # check number of channels
            if (C > 3):
                print("WARNING: images with more than 3 channels are currently not support. using only first 3 channels")
                C = 3
                img = img[0:3,...]

            if (convertToGrayscale):
                C = 1

            # close archive if opened
            if (zip is not None):
                zip.close()

            openedArchive = None
            zip = None

        # make sure C H W are respectively > 0 by now
        assert(C > 0), "ERROR: number of input image channels has to be > 0"
        assert(W > 0), "ERROR: number of input image width has to be > 0"
        assert(H > 0), "ERROR: number of input image height has to be > 0"

        print("input N E V C H W : N=%d, E=%d, V=%d, C=%d, H=%d, W=%d" %(N,E,V,C,H,W))

        # override width height with values given in output paramater
        W = output_resolution[0]
        H = output_resolution[1]
        assert(W > 0), "ERROR: number of input image width has to be > 0"
        assert(H > 0), "ERROR: number of input image height has to be > 0"

        print("N E V C H W for h5 file: (%d, %d, %d, %d, %d, %d)" %(N,E,V,C,H,W))


        # write aperture masks
        if mask_imgs is not None:

            print("aperture masks shape : ", mask_imgs.shape)
            resized_masks = resizeImage(mask_imgs, output_resolution)
            print("aperture masks shape resized : ", resized_masks.shape)

            
            if (resized_masks.shape[2] != C):
                print("aperture masks channels does not fit image stack channels. correcting shape....")
                if (C < resized_masks.shape[2]):
                    resized_masks = resized_masks[:,:,0:C,...]
                else:
                    resized_masks = np.repeat(resized_masks[:,:,0:1,...], C, 2) # repeat along channels axis
                print("aperture masks shape corrected : ", resized_masks.shape)
            

            # check if aperture masks fit to configuration of Eyes/Views/Channels/Height/Width
            assert(E == resized_masks.shape[0]), "ERROR: number of used eyes has to match aperture mask eye dimensions"
            assert(V == resized_masks.shape[1]), "ERROR: number of used view has to match aperture mask views"
            assert(C == resized_masks.shape[2]), "ERROR: number of image channels has to match aperture mask channels"
            assert(H == resized_masks.shape[3]), "ERROR: number of image width has to match aperture mask width"
            assert(W == resized_masks.shape[4]), "ERROR: number of image height has to match aperture mask height"

            h5_file.create_dataset("aperture_masks", resized_masks.shape, dtype=resized_masks.dtype, data=resized_masks)
        else:
            print("WARNING : h5 file will contain NO aperture masks !!!")

        # iterate over all samples
        print("Writing images....be patient")
    
        imagedataset = None # initialize h5 dataset for images
        
        regionmask_dataset = None # initialize h5 dataset for regionmaks

        # initialize dataset samples index : this is different from sampleIdx since sampleIdx is an index in the list of all samples (training+test) !!
        datasetSampleIndex = 0

        for sampleIdx in sampleIndices:
        
            print("   image %d / %d" %(datasetSampleIndex+1, len(sampleIndices)))

            # get archive path of current image
            archivename = str(labels['archive'][sampleIdx])
            
            archivepath = archivename
            if (os.path.exists(archivepath) == False):
                archivepath = os.path.join(footagepath, archivepath)

            # open required archive if it is not yet opened
            if (openedArchive != archivepath):
                # close archive if opened
                if (zip is not None):
                    zip.close()
                # open archive
                zip = zipfile.ZipFile(archivepath, 'r')
                openedArchive = archivepath
           
            # allocate memory for image
            img = np.zeros((E,V,C,H,W), dtype=np.uint8)
            
            # allocate memory for region maps if available
            if (use_region_masks):
                converted_regionmask = np.zeros((E,V,C,H,W), dtype=np.uint8)


            # iterate over eyes
            currentEyeDim = 0
            for eyeIdx in range(2):
            
                if (use_region_masks):
                    regionmask_withskin = None
                    regionmask_withoutskin = None

                numViews = 0
                viewLabels = list()
                if (eyeIdx == 0): #left eye
                    if (numViewsLeft > 0):
                        numViews = numViewsLeft
                        viewLabels = eye_view_labels_left
                else: # right eye
                    if (numViewsRight > 0):
                        numViews = numViewsRight
                        viewLabels = eye_view_labels_right
                
                if (numViews == 0): # skip eye if there are now views for it
                    continue

                viewsWritten = 0

                # iterate over views of current eye
                #for viewIdx in range(numViews):
                for viewName in viewLabels:
                    
                    viewNameSplit = viewName.split("_")
                    # process image
                    if (viewNameSplit[0] == "image"):
                        viewIdx = int(viewNameSplit[-1])
                        #print("handle %s with view index: %d" %(viewName, viewIdx))
                        #print(labels[viewName][sampleIdx])
                        imagename = np.array2string(labels[viewName][sampleIdx])[1:-1]

                        singleViewImage = iio.imread(BytesIO(zip.open(imagename, 'r').read()))

                        # make sure image file is in CHW, (if WH only convert to CHW)
                        if (len(singleViewImage.shape) == 2):
                            singleViewImage = np.expand_dims(singleViewImage, 0)
                            #print("expanding image dimension : ", singleViewImage.shape)
                        assert (len(singleViewImage.shape) == 3), "ERROR: unknown format of image !"

                        if (singleViewImage.shape[0] > 4):
                            #print("tranpose required for image file : ", imagename)
                            singleViewImage = np.transpose(singleViewImage, [2, 0, 1])

                        # check number of channels
                        if (singleViewImage.shape[0] > 3):
                            #print("WARNING: images with more than 3 channels are currently not support. using only first 3 channels")
                            singleViewImage = singleViewImage[0:3,...]

                        #pdb.set_trace()

                        if (convertToGrayscale and singleViewImage.shape[0] > 1):
                            type = singleViewImage.dtype
                            singleViewImage = np.sum(np.asarray(singleViewImage, np.float), 0, np.float, None, np.True_) / float(singleViewImage.shape[0])
                            singleViewImage = np.asarray(singleViewImage, type)

                        # resize image to output resolution
                        singleViewImageResized = resizeImage(singleViewImage, output_resolution)
                        #print("resized image shape : ", singleViewImageResized.shape)
                    
                        # write image
                        img[currentEyeDim,viewIdx,...] = singleViewImageResized # fill image for E V C H W

                    # process region map
                    if (use_region_masks):
                        if (viewNameSplit[0] == "regionmask"):
                            viewIdx = int(viewNameSplit[-1])
                            #print("handle regionmaks with view index: ", viewIdx)
                            maskfilename = np.array2string(labels[viewName][sampleIdx])[1:-1]
                            singleRegionMaskImage = iio.imread(BytesIO(zip.open(maskfilename, 'r').read()))
                            # make sure image file is in CHW
                            assert (len(singleRegionMaskImage.shape) == 3), "ERROR: region masks has to be provided as 3-channel image !"
                            singleRegionMaskImage = np.transpose(singleRegionMaskImage, [2, 0, 1])
                            #print("shape when loading singleRegionMaskImage :", singleRegionMaskImage.shape)

                            if (viewNameSplit[1] == "withskin"):
                                if (regionmask_withskin is None):
                                    regionmask_withskin = np.zeros((V,
                                                                    singleRegionMaskImage.shape[0], # channels
                                                                    singleRegionMaskImage.shape[1], # height
                                                                    singleRegionMaskImage.shape[2]), # width
                                                                    dtype=np.uint8)
                                    #print('init region mask with skin with shape ', regionmask_withskin.shape)
                                regionmask_withskin[viewIdx,...] = singleRegionMaskImage # fill image for V C H W

                            if (viewNameSplit[1] == "withoutskin"):
                                if (regionmask_withoutskin is None):
                                    regionmask_withoutskin = np.zeros((V,
                                                                       singleRegionMaskImage.shape[0], # channels
                                                                       singleRegionMaskImage.shape[1], # height
                                                                       singleRegionMaskImage.shape[2]), # width
                                                                       dtype=np.uint8)
                                    #print('init region mask with skin without shape ', regionmask_withoutskin.shape)
                                regionmask_withoutskin[viewIdx,...] = singleRegionMaskImage # fill image for V C H W
                
                
                # convert region masks 
                if (use_region_masks):
                    for viewIdx in range(numViews):
                        
                        ap_mask = None # handle no aperture mask case
                        if (mask_imgs is not None): # if aperture masks exist get copy
                            ap_mask = mask_imgs[currentEyeDim,viewIdx,...]

                        regionmaskSingleView = generateRegionMap(regionmask_withskin[viewIdx,...], regionmask_withoutskin[viewIdx,...], ap_mask)

                        # generate semantic labels out of region mask
                        o1, o2, o3, o4, o5, o6, o7, o8 = generateSemanticLabels(regionmaskSingleView)
                        labels['float_pupil_occlusion_ratio'][sampleIdx] = o1
                        labels['float_unoccluded_pupil_center_x'][sampleIdx] = o2
                        labels['float_unoccluded_pupil_center_y'][sampleIdx] = o3
                        labels['float_unoccluded_pupil_ellipse_pos_x'][sampleIdx] = o4
                        labels['float_unoccluded_pupil_ellipse_pos_y'][sampleIdx] = o5
                        labels['float_unoccluded_pupil_ellipse_axis_0'][sampleIdx] = o6
                        labels['float_unoccluded_pupil_ellipse_axis_1'][sampleIdx] = o7
                        labels['float_unoccluded_pupil_ellipse_tilt'][sampleIdx] = o8

                        #print("region mask single view : ", regionmaskSingleView.shape)
                        regionmaskSingleViewResized = resizeImage(regionmaskSingleView, output_resolution)
                        #print("region mask single view resized : ", regionmaskSingleViewResized.shape)
                        converted_regionmask[currentEyeDim, viewIdx, ...] = regionmaskSingleViewResized

                currentEyeDim += 1

            # initialize empty image dataset if not existing yet
            if imagedataset is None :
                imagedataset = h5_file.create_dataset("images", (N, E, V, C, H, W), dtype=img.dtype)
                print("N E V C H W for h5 file (images): (%d, %d, %d, %d, %d, %d)" %(N,E,V,C,H,W))

            # write image to h5 file
            imagedataset[datasetSampleIndex] = img
            
            # write region masks to h5 file (if activated)
            if (use_region_masks):
                # initialize empty dataset if not existing yet
                if regionmask_dataset is None :
                    regionmask_dataset = h5_file.create_dataset("region_maps", (N, E, V, C, H, W), dtype=converted_regionmask.dtype)
                    print("N E V C H W for h5 file (region masks): (%d, %d, %d, %d, %d, %d)" %(N,E,V,C,H,W))

                if (converted_regionmask.shape[2] != C):
                    print("region map shape : ", converted_regionmask.shape)
                    print("region map channels does not fit image stack channels. correcting shape....")
                    if (C > converted_regionmask.shape[2]):
                        converted_regionmask = np.repeat(converted_regionmask, C, 2) # repeat along channels axis
                    print("region map shape corrected : ", resized_masks.shape)
                            # check if aperture masks fit to configuration of Eyes/Views/Channels/Height/Width
                
                assert(E == converted_regionmask.shape[0]), "ERROR: number of used eyes has to match region map eye dimensions"
                assert(V == converted_regionmask.shape[1]), "ERROR: number of used view has to match region map views"
                assert(C == converted_regionmask.shape[2]), "ERROR: number of image channels has to match region map channels"
                assert(H == converted_regionmask.shape[3]), "ERROR: number of image width has to match region map width"
                assert(W == converted_regionmask.shape[4]), "ERROR: number of image height has to match region map height"

                # write region mask
                regionmask_dataset[datasetSampleIndex] = converted_regionmask

            
            # increment dataset samples index : this is different from sampleIdx since sampleIdx is an index in the list of all samples (training+test) !!
            datasetSampleIndex = datasetSampleIndex + 1

        print('number of frames written for dataset : ', datasetSampleIndex)

        # close archive if opened
        if (zip is not None):
            zip.close()

        # write labels to H5
        print("Writing label info for selected samples...")
        for labelName in labellist:
            print("writing label ", labelName)
            labelSubSet = np.array(labels[labelName][sampleIndices], dtype='S') # convert object to string with dynamic length: see https://docs.h5py.org/en/stable/strings.html
            h5_file.create_dataset(labelName, labelSubSet.shape, dtype=labelSubSet.dtype, data=labelSubSet)



# Create the h5 file
# dataset is a zipclip dataset with clips(full_path_to_zip_archive, label_list of list(dict) e.g. [ {"val" : 5}, {"val": 2.5}, etc. ] )
# full out path is the out path of the h5 file. Multiple h5 files will be created for train and test, and these will have "train" and "test" appended accordingly.
# args contains args.subsample and args.eye. it is a legacy parameter that will be deleted.
# output resolution is self explanatory. In our new scheme, it is the only required pre-training augmentation.

# Create the H5 file, called from DataManagementApp.py, but usable as a
# standalone function
# All paths are absolute.
#def export_h5_dense(csvfile, full_out_path, args, output_resolution = (255, 191), use_masks = True):
def createH5(csvfile, h5outputpath, output_resolution, sampleRatio=None, labeledSampling=None, use_aperture_masks=True, use_region_masks=False, eyes = 'monocular', convertToGrayscale=False):
    starttime = datetime.datetime.now()

    print ('use aperture masks : ', str(use_aperture_masks))
    print ('use region masks : ', str(use_region_masks))
    print ('sampleRatio : ', sampleRatio)
    print ('labeledSampling : ', labeledSampling)

    # extract labels from CSV file
    (labels, labellist, numSamples, validLabels) = parseCSV(csvfile)

    # print all labels extracted directly from footage
    print('Labels extracted from footage :', labellist)
    
    # check if labels for labeled sampling are contained in labellist
    if (sampleRatio > 0 and labeledSampling):
        for l in labeledSampling:
            if (l not in labeledSampling):
                print("ERROR: %s for sampling is not contained in provided labels !" %(l))
                return


    # initialize semantic labels if using region masks
    if use_region_masks:

        print('Semantic labels will be generated using region masks and added to dataset.')

        # all the semantic labels to be formed using region masks
        semantic_label_names = [
            'float_pupil_occlusion_ratio',
            'float_unoccluded_pupil_ellipse_pos_x',
            'float_unoccluded_pupil_ellipse_pos_y',
            'float_unoccluded_pupil_ellipse_axis_0',
            'float_unoccluded_pupil_ellipse_axis_1',
            'float_unoccluded_pupil_ellipse_tilt',
            'float_unoccluded_pupil_center_x',
            'float_unoccluded_pupil_center_y'
        ]

        # add semantic label names to labellist, and initialize fields for them in labels
        for semantic_label_name in semantic_label_names:
            labellist.append(semantic_label_name)
            labels[semantic_label_name] = np.zeros(labels['archive'].shape)
        
        # print all labels
        print('All labels :', labellist)



    # get footage path
    footagepath = os.path.dirname(csvfile)
    
    # load aperture masks
    mask_imgs = loadApertureMasks(footagepath, use_aperture_masks)
    
    # create indices for training + test data set
    if sampleRatio is not None and (sampleRatio > 0):
        assert (sampleRatio <= 0.5), "Sampling ratio of training/testing has to be <= 0.5 (50 % training data / 50 % test data)"
        
        print('creating a h5 files for training and training data with sample ratio ', sampleRatio)

        # write two h5 files for training and testing
        numTestSamples = int(numSamples * sampleRatio)
               
        trainingDataIndices = np.arange(numSamples) # initialize list containing all samples for training

        # generate attribute vector based on labels passed by the user.
        attributeForSampling = None
        if (labeledSampling):
            # Create an attribute vector that is going to hold the values of the labels of interest.
            attributeForSampling = []

            for labelTitle in labeledSampling:
                assert(labelTitle in labellist), 'Label %s is not in the dataset. Please check the label name.'%labelTitle

                # pdb.set_trace()
                attributeForSampling.append(labels[labelTitle])

        # generate list of test samples
        testDataIndices = generateSampleIndices(numTestSamples, numSamples, attributeForSampling) # generate list of test samples

        # remove test samples from training samples
        trainingDataIndices = np.array(list(set(trainingDataIndices) - set(testDataIndices)))

        # remove all samples that have the same combinations of given labels
        if labeledSampling is not None:
            print('removing all samples in training that have similar labels in test set')
            indicesFromLabeledSampling = np.zeros([numSamples, len(labeledSampling)], dtype=np.uint8)

            print(labeledSampling)

            for attrIndex, attr in enumerate(labeledSampling):
                #print('attrIndex = ', attrIndex)
                #print('attr = ', attr)
                for trainIndex in testDataIndices:
                    #print(trainIndex)
                    value = labels[attr][trainIndex]
                    #print(value)
                    for allSamplesIndex in range(numSamples):
                        testvalue = labels[attr][allSamplesIndex]
                        #print(testvalue)
                        if testvalue  == value:
                            #print('values match at full index ', allSamplesIndex)
                            indicesFromLabeledSampling[allSamplesIndex][attrIndex] = 1
            
            #print(indicesFromLabeledSampling)

            indicesToBeRemoved = []
            for allSamplesIndex in range(numSamples):
                if np.sum(indicesFromLabeledSampling[allSamplesIndex]) == len(attributeForSampling):
                    indicesToBeRemoved.append(allSamplesIndex)

            print('test indices : ', testDataIndices)
            print('indices to be removed : ', indicesToBeRemoved)
            print('number training values before removal : ', len(trainingDataIndices))
            trainingDataIndices = np.array(list(set(trainingDataIndices) - set(indicesToBeRemoved)))
            print('number training values after removal : ', len(trainingDataIndices))


        # write training h5 file containing training samples
        trainingH5FilePath = h5outputpath
        trainingH5FilePath = trainingH5FilePath.replace(".h5", "_train.h5")
        print('Creating h5 for training data', trainingH5FilePath)
        writeH5File(trainingH5FilePath, labels, labellist, trainingDataIndices, footagepath, output_resolution, mask_imgs, eyes, use_region_masks, convertToGrayscale )
        print('done with h5 training data file : ' +  trainingH5FilePath + '\n')

        # write test h5 file containing test samples
        testH5FilePath = h5outputpath
        testH5FilePath = testH5FilePath.replace(".h5", "_test.h5")
        print('Creating h5 for test data : ', testH5FilePath)
        writeH5File(testH5FilePath, labels, labellist, testDataIndices, footagepath, output_resolution, mask_imgs, eyes, use_region_masks, convertToGrayscale )
        print('done with h5 test data file : ' +  testH5FilePath + '\n')

    else:
        # write single h5 file containing all samples
        print('creating a single h5 file containing all samples : ', h5outputpath)
        trainingDataIndices = np.arange(numSamples)
        writeH5File(h5outputpath, labels, labellist, trainingDataIndices, footagepath, output_resolution, mask_imgs, eyes, use_region_masks, convertToGrayscale )
        print('done with h5 file : ' +  h5outputpath + '\n')
    
    endtime = datetime.datetime.now()
    elapsedtime = endtime - starttime
    print("H5 file created in ", elapsedtime.seconds," seconds\n\n") 

#----------------------------------------------------------------------------
	
