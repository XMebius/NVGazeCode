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

"""
Eye annotation tool
drag and drop h5 file into GUI
and start annotating
"""
from PIL import Image
import numpy as np
import h5py

################################################################################################
# function to resize single channel image 
# Args:
#   img .. numpy array in the format [H,W]
#   size .. tupel of resized image resolution in the form (W,H) (e.g. (640, 480))
################################################################################################
def resizeImgChannel(img, size, resizeMethod = 'bicubic'):

    assert (resizeMethod == 'bicubic' or resizeMethod == 'nearest'), "Image resize method has to be 'bicubic' or 'nearest'"

    interpolationMethod = Image.BICUBIC
    if (resizeMethod == 'nearest'):
        interpolationMethod = Image.NEAREST

    pilImg = Image.fromarray(img)
    pilImg = pilImg.resize(size, interpolationMethod)
    resizedArray =  np.asarray(pilImg, dtype=img.dtype, order='C')
    return resizedArray

################################################################################################
# recursive function to resize images/images batches into the given resolution
#
# Args:
#   img .. [batches of] image[s] : [n1,n2,..,C,]H,W
#   output_resolution .. tupel of resized image resolution in the form (W,H) (e.g. (640, 480))
#   resizeMethod .. 'bicubic' or 'nearest'
################################################################################################
# expected format is 
def resizeImage(img, output_resolution, resizeMethod = 'bicubic'):

    imgDimensions = len(img.shape)
    assert len(img.shape) > 1, "one-dimensional arrays not supported for image resize !"
    if len(img.shape) == 2:
        return resizeImgChannel(img, output_resolution, resizeMethod)
    return np.asarray([resizeImage(img_one_less_dim, output_resolution, resizeMethod) for img_one_less_dim in img])
  

######################################################################################
# convert string to h5 dataype (numpy compatible)
######################################################################################
def getH5Datatype(s):
    
    stringdt = h5py.special_dtype(vlen=str) # h5f5 data type for strings with variable length

    conversionSuccessful = False
    dt = stringdt

    s = s.strip()

    if (s == 'string'):
        dt = stringdt
        conversionSuccessful = True
    elif (s == 'double'):
        dt = np.float64
        conversionSuccessful = True
    elif (s == 'float'):
        dt = np.float32
        conversionSuccessful = True
    elif (s == 'int'):
        dt =  np.int32
        conversionSuccessful = True
    elif (s == 'bool'):
        dt =  np.bool_
        conversionSuccessful = True
    else:
        dt = stringdt
        # conversion NOT successful in this case, assume string

    return (dt, conversionSuccessful)

# function assumes CHW
def convertImageToGrayscale(img):
    
    sourceDataType = img.dtype

    # convert to float32
    imgFloat = np.asarray(img, dtype=np.float32)

    imgGrayScale = (imgFloat[0] + imgFloat[1] + imgFloat[2]) / 3.0
    return np.asarray(imgGrayScale, dtype=sourceDataType)

# function assumes CHW
def convertGrayscaleToRGB(img):
    return np.vstack((img,img,img))
