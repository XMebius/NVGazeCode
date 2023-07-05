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

import json, pdb, glob, os, sys, shutil, datetime
import pandas as pd
import numpy as np
import random
from data_prep import * # actual heavy-lifting of dataset generation
import h5py
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../.")
import utils

def load_h5file(filename):    

    h5_file_handler = None
    imagePtr = None
    labels = None
    acceptedLabels = None
    rejectedLabels = None

    print('\nopening h5 file')
    h5_file_handler = h5py.File(filename, 'r')
    
    if 'validSamples' in h5_file_handler.keys():
        acceptedLabels = h5_file_handler['validSamples'][:]
    else:
        print('     WARNING : no info about accepted samples')

    if 'rejectedSamples' in h5_file_handler.keys():
        rejectedLabels = h5_file_handler['rejectedSamples'][:]
    else:
        print('     WARNING : no info about rejected samples')

    
    for key in h5_file_handler.keys():
        if (labels is None):
            labels = dict()
        labels[key] = h5_file_handler[key][:]

    # read images
    imagePtr = h5_file_handler['images'] # type : h5 dataset

    return h5_file_handler, imagePtr, labels, acceptedLabels, rejectedLabels

def closeH5File(h5_file_handler):
    if (h5_file_handler is not None):
        print('closing h5 file\n\n')
        h5_file_handler.close()


if __name__ == "__main__":


    print("\n\n=====================\nMERGE DATABASE SCRIPT\n=====================\n\n")

    databasefiles = [
        "C:/NvGaze/Datasets/h5/sampleSyntheticDataset_1.h5",
        "C:/NvGaze/Datasets/h5/sampleSyntheticDataset_2.h5",
        ]

    databaseratio = [
        1.0, # full first dataset
        0.5, # half of second dataset
        ]

    requiredLabels = [
        'float_pupil_x',
        'float_pupil_y',
        'float_pupilellipse_minor',
        'float_pupilellipse_major',
        'float_pupilellipse_angle',
        ]

    # specify whether to skip rejected samples -> requires dataset with knowledge of rejected samples
    # use LabelAnnoationTool if you want to provide this data
    # array size has to match number of input datasets
    skipRejectedSamples = [
        False, 
        False,
        ]

    # specify whether to skip nonaccepted samples
    # use LabelAnnoationTool if you want to provide this data
    # array size has to match number of input datasets
    skipNonAcceptedSamples = [
        False,
        False,
        ]
        
    convert_to_grayscale            = True

    randomize_input_samples         = True
    randomize_merged_samples        = True

    write_test_train_dataset        = True     # in addition to full dataset also write split train/test dataset
    test_train_ratio                = 0.2      # e.g 0.2 test data remaining is training data

    eyes                    = 'monocular'
    eyeSelection            = 'L' # L or R
    #eyeSelection            = 'R' # L or R
    
    # images will be resized to this resolution
    # pupil center labels will also be recomputed
    # ATTENTION: ellipse data is currently not handled on resize
    output_resolution       = [293,293] 
    
    out_drive_path = 'C:/NvGaze/Datasets/h5/' # path in which combined dataset will be stored
    merged_dataset_name = "combinedSyntheticDataset_293.h5" # file name for combined dataset

    # check whether the output dataset already exists and get new name by adding/increasing index
    nameToBeTested = merged_dataset_name

    merged_dataset_name = os.path.join(out_drive_path, nameToBeTested)
    print("h5 file to be created : %s\n"%(merged_dataset_name))

    success = True

    selectionPerDatabase = []

    for databaseidx, databasefile in enumerate(databasefiles):
        
        print("analyze dataset ", databasefile)
        
        # open h5 file and check for requested labels
        (h5_file_handler, imagePtr, labels, acceptedSamples, rejectedSamples) = load_h5file(databasefile)

        # check for images
        if (imagePtr is None):
            print('Abort. No images in database %s !\n'%(databasefile))
            success = False
            break

        # check for requested labels
        for l in requiredLabels:
            if (l not in labels):
                print('Abort. Requested label %s not in database %s !\n'%(l, databasefile))
                success = False
                break

        # generate samples for this dataset
        numSamplesImages = imagePtr.shape[0]
        numSamplesLabels = labels[requiredLabels[0]].shape[0]
        if (numSamplesImages != numSamplesLabels):
                print('Abort since image and label count are different !\n')
                success = False
                break

        # get all samples
        allSampleIndices = list(range(numSamplesImages))
        selectedSamples = allSampleIndices.copy()

        if (skipNonAcceptedSamples[databaseidx]):
            if (acceptedSamples is None):
                print('Abort since information about accepted labels is required but not provided !\n')
                success = False
                break
            else:
                selectedSamples = acceptedSamples
        else:
            # remove rejected samples if required
            if (skipRejectedSamples[databaseidx]):
                if (rejectedSamples is None):
                    print('Abort since information about rejected labels is required but not provided !\n')
                    success = False
                    break
                else:
                    print('skipping rejected samples')
                    pdb.set_trace()
                    for idx in rejectedSamples:
                        selectedSamples.remove(idx)
        
        # randomize if shuffling is active
        if (randomize_input_samples):
            random.shuffle(selectedSamples)

        print('num all samples      : %d'%(len(allSampleIndices)))
        if rejectedSamples is not None:
            print('num rejected samples : %d'%(len(rejectedSamples)))
        print('num valid samples    : %d'%(len(selectedSamples)))

        # select given ratio
        numSelectedSamples = int(len(selectedSamples) * databaseratio[databaseidx])
        selectedSamples = selectedSamples[0:numSelectedSamples]
        print('num selected samples : %d'%(len(selectedSamples)))
        
        # save selected samples for this database
        selectionPerDatabase.append(selectedSamples.copy())
                
        closeH5File(h5_file_handler)

    if (success == False):
        print('\n\nERROR: Abort.\n')
    else:

        # merge databases using previously computed selections
        print('Merging databases with following sampling ratios:')
        numCombinedImageSamples = 0
        for databaseidx, databasefile in enumerate(databasefiles):
            print('     %d samples from %s'%(len(selectionPerDatabase[databaseidx]), databasefile))
            numCombinedImageSamples += len(selectionPerDatabase[databaseidx])
        print('     %d total combined samples'%(numCombinedImageSamples))
        print('\nOutput file %s\n'%(merged_dataset_name))
        

        # ask before writing merged dataset
        answer = input("Continue ? (type 'y') : ")
        if (answer != 'y' and answer != 'Y'):
            print('User abort process\n')
            success = False

        if (success):
            # randomize output samples if required
            outputSampleIndices = list(range(numCombinedImageSamples))
            if (randomize_merged_samples):
                random.shuffle(outputSampleIndices)
            indices = outputSampleIndices.copy()

            # write h5 file
            outputfile = merged_dataset_name
            print('\n\nwrite %s..\n'%(outputfile))
            with h5py.File(outputfile, 'w') as h5_file:
                
                # create datasets for labels in h5 file
                out_label = dict()
                for l in requiredLabels:
                    out_label[l] = h5_file.create_dataset(l, [len(indices)], dtype=np.float32)

                numSamplesUsedFromThisDataset = 0

                # create image dataset
                imageDatasetshape = [len(indices), 1, 1, 1, output_resolution[1], output_resolution[0]]
                print('datasetshape ', imageDatasetshape)
                imagedataset = h5_file.create_dataset("images", imageDatasetshape, dtype=np.uint8)

                # running start index for datasets, updated after one dataset is done
                startIndexForOutput = 0

                # collect input images
                for databaseidx, databasefile in enumerate(databasefiles):
                    print("reading from dataset ", databasefile)
                    # open h5 file and check for requested labels
                    (h5_file_handler, imagePtr, labels, acceptedSamples, rejectedSamples) = load_h5file(databasefile)

                    samples = selectionPerDatabase[databaseidx]

                    inputResolution = (imagePtr.shape[-1], imagePtr.shape[-2])

                    resizeRequired = False
                    if (inputResolution[0] != output_resolution[0] or inputResolution[1] != output_resolution[1]):
                        resizeRequired = True
                    print("\n INFO: Resize of images and labels is performed ! \n")

                        
                    print('writing images to h5 file....')
                    # get images, if required resize
                    for inDataSetIndex, s in enumerate(samples):
                        image = imagePtr[s,0]
                        if (resizeRequired):
                            image = utils.resizeImage(image, output_resolution)
                            
                        imagedataset[indices[startIndexForOutput + inDataSetIndex],...] = image

                    print('writing labels to h5 file....')
                    for l in requiredLabels:
                        for inDataSetIndex, s in enumerate(samples):

                            value = labels[l][s]
                            if (l == 'float_pupil_x' and resizeRequired):
                                value = float(value) / inputResolution[0] * output_resolution[0]
                            if (l == 'float_pupil_y' and resizeRequired):
                                value = float(value) / inputResolution[1] * output_resolution[1]
                            if (l == 'float_pupilellipse_major' and resizeRequired):
                                value = float(value) / inputResolution[0] * output_resolution[0]
                            if (l == 'float_pupilellipse_minor' and resizeRequired):
                                value = float(value) / inputResolution[1] * output_resolution[1]
                                
                            out_label[l][indices[startIndexForOutput + inDataSetIndex]] = value

                    startIndexForOutput += len(samples)
                    
                    # close dataset
                    closeH5File(h5_file_handler)

            # write train/test dataset
            if (write_test_train_dataset):

                # split merged dataset into test and train dataset using given ratio
                numTestSamples = int(test_train_ratio * len(outputSampleIndices))
                testSampleIndices  = list(range(numTestSamples))

                numTrainSamples = len(outputSampleIndices) - numTestSamples
                        
                trainH5Filename = outputfile.replace('.h5', '_train.h5')
                testH5Filename = outputfile.replace('.h5', '_test.h5')

                (h5_file_handler, imagePtr, labels, acceptedSamples, rejectedSamples) = load_h5file(merged_dataset_name)

                print("\n\nwriting test+train h5 file\n")
                print("     ", trainH5Filename)
                print("     ", testH5Filename)

                numSamples = imagePtr.shape[0]

                with h5py.File(trainH5Filename, 'w') as trainH5:

                    # create datasets for labels in h5 file
                    out_label = dict()
                    for l in requiredLabels:
                        out_label[l] = trainH5.create_dataset(l, [numTrainSamples], dtype=np.float32)

                    # create image dataset
                    imageDatasetshape = [numTrainSamples, 1, 1, 1, output_resolution[1], output_resolution[0]]
                    print('datasetshape ', imageDatasetshape)
                    imagedataset = trainH5.create_dataset("images", imageDatasetshape, dtype=np.uint8)

                    for idx in range(numTrainSamples):

                        # write image
                        imagedataset[idx] = imagePtr[idx]

                        # write all labels
                        for l in requiredLabels:
                            out_label[l][idx] = labels[l][idx]


                with h5py.File(testH5Filename, 'w') as testH5:

                        # create datasets for labels in h5 file
                        out_label = dict()
                        for l in requiredLabels:
                            out_label[l] = testH5.create_dataset(l, [numTestSamples], dtype=np.float32)

                        # create image dataset
                        imageDatasetshape = [numTestSamples, 1, 1, 1, output_resolution[1], output_resolution[0]]
                        print('datasetshape ', imageDatasetshape)
                        imagedataset = testH5.create_dataset("images", imageDatasetshape, dtype=np.uint8)

                        for idx in range(numTestSamples):

                            # write image
                            imagedataset[idx] = imagePtr[idx + numTrainSamples]

                            # write all labels
                            for l in requiredLabels:
                                out_label[l][idx] = labels[l][idx+numTrainSamples]
       
            print('\n\n all done!')
    
    print("\n\n=====================\nEND OF SCRIPT\n=====================\n\n")
