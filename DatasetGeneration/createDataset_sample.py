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

from data_prep import * # actual heavy-lifting of dataset generation
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../.")
from utils import searchPathRecursivelyForString, createCombinedFootageFile
    

if __name__ == "__main__":

    datasetlist = [
        "C:/NvGaze/Datasets/raw/nvgaze_male_01",
        ]

    aperture_masks  = False
    region_masks    = False

    set_description        = 'sampleSyntheticDataset'

    #sampleRatio = 0.2 # > 0 means create train+test
    sampleRatio = 0.0 # > 0 means create train+test

    combinedfootagepath     = 'C:/NvGaze/Datasets/temp/'
    
    convert_to_grayscale    = True
    eyes                    = 'monocular'

    output_resolution       = [320,240] # aspect ratio needs to match source data !
    
    labeledSampling         = None

    success = True

    # figure out the path to the dataset
    dataset_output_path = 'C:/NvGaze/Datasets/h5/'

    if (os.path.exists(dataset_output_path) == False):
        os.mkdir(dataset_output_path)

    print("dataset description", set_description)
    print("usedDataSets", datasetlist)
    print("len(datasetlist)", len(datasetlist))

    # check whether the filename already exists
    dst_setname = set_description + '.h5'
    try_cnt = 1
    while searchPathRecursivelyForString(dataset_output_path, dst_setname.replace('.h5','')): # if filename already exists
        dst_setname = set_description + '_' + str(try_cnt) + '.h5'
        try_cnt += 1
    print("h5 file to be created : ", dst_setname)

    # create combined csvfile holding labels of all selected datasets
    csvfilelist = []
    for setpath in datasetlist:
        singlesetcsv = os.path.join(setpath, 'footage_description.csv')
        csvfilelist.append(singlesetcsv)

    if (os.path.exists(combinedfootagepath) == False):
        os.mkdir(combinedfootagepath)

    outputfile = os.path.join(combinedfootagepath, 'footage_description.csv')
    csvfile, writing_successful = createCombinedFootageFile(csvfilelist, outputfile)
    if (writing_successful == False):
        print("Error on writing combined csv file")
        success = False

    if (success):

        print("used csv file : ", csvfile)
        print("convert_to_grayscale : ", convert_to_grayscale)

        # @param path to csvfile, path to h5output, output resolution, sample ratio to create test set, labeled sampling, use of mask, eye.
        createH5(
            csvfile,
            h5outputpath = os.path.join(dataset_output_path, dst_setname),
            output_resolution = output_resolution,
            sampleRatio = sampleRatio,
            labeledSampling = labeledSampling,
            use_aperture_masks = aperture_masks,
            use_region_masks = region_masks,
            eyes = eyes,
            convertToGrayscale = convert_to_grayscale
            )

