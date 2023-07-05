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

import os, os.path, glob, sys
from scipy import misc
from io import BytesIO
import numpy as np
from PIL import Image
import random
import numpy as np
import h5py

import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

def writeH5File(h5filepath, image_list):

    print("writing h5 file : ", h5filepath)

    with h5py.File(h5filepath, 'w') as h5_file:

        for i in range(len(image_list)):

            # create one dataset for each image since resolution might be different
            datasetname = 'image_' + str(i)
            imagedataset = h5_file.create_dataset(datasetname, (image_list[i].shape[0], image_list[i].shape[1]), dtype=img.dtype)
            imagedataset[...] = image_list[i]
        
        print('%d images written to h5 file.'%(len(image_list)))


if __name__ == '__main__':
    
    print('loading reflection images...')

    reflectionImagesRootFolder = 'C:/NvGaze/Datasets/reflections/'
    reflectionDatasetOutputFile = 'C:/NvGaze/Datasets/h5/reflectiondataset.h5'

    if (os.path.exists(reflectionDatasetOutputFile)):
        print("WARNING: file %s exists already. Please remove file and restart dataset generation"%(reflectionDatasetOutputFile))
        exit()

    maxNumImages = 500
    
    minImageWidth = 320     # should be larger than images used later in training
    minImageHeight = 320    # should be larger than images used later in training

    limitNumImages = True
    randomizeImages = True

    #Retrieve all files from listed directories. We will discard them on load if they are not images.
    imageNames = []
    reflectionImageFolders = []

    extensions = [".png", ".jpg", ".jpeg"]
    imageNames = [f for f in glob.glob(os.path.join(reflectionImagesRootFolder, "**/*"), recursive=True) if os.path.splitext(f)[1].lower() in extensions]

    if (len(imageNames) == 0):
        print("ERROR: no image files found in %s"%(reflectionImagesRootFolder))
        print("Exiting")
        exit()

    numImages = len(imageNames)
    print("%d files found"%(numImages))

    if (limitNumImages):
        numImages = min(numImages, maxNumImages)

    if (randomizeImages):
        random.shuffle(imageNames)
    
    reflectionImages = []
    numAppendedImages = 0
    index = 0
    while (numAppendedImages < numImages and index < len(imageNames)):
        reflectionImagePath = imageNames[index]
        print('loading image %d / collected %d / target %d: %s'%(index+1, numAppendedImages, numImages, reflectionImagePath))
        try:
            img = np.asarray(Image.open(reflectionImagePath))
            addThisImage = False
            if (len(img.shape) > 1 and img.shape[0] >= minImageWidth and img.shape[1] >= minImageHeight):
                img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) # convert to gray
                if (len(img_gray.shape) > 1):
                    addThisImage = True
                    reflectionImages.append(img_gray)
                    numAppendedImages = numAppendedImages + 1
            if not addThisImage:
                print('WARNING: skipping image %d due to insufficient shape'%(index))
                print(img.shape)

        except Exception:
            print("can't open the following reflection image file :" + reflectionImagePath)
            
        index = index + 1

    print('%d reflection images loaded...'%(len(reflectionImages)))

    writeH5File(reflectionDatasetOutputFile,reflectionImages)
