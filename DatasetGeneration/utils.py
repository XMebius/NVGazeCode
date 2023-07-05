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

import os, os.path, glob, zipfile, pdb, math, sys, platform
import shutil
from scipy import misc
import zipfile
from io import BytesIO
import numpy as np
from PIL import Image
import random
import numpy as np
import h5py
import csv
import torch


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
# calculate Euclidean distance between two numpy arrays.
######################################################################################
def Euclidean_dist(a, b, axis = None):
    return np.linalg.norm(a - b, axis = axis)

######################################################################################
# calculate resolution that results in no loss in information when there is no padding.
######################################################################################
def get_padding_free_resolution(strides = (2,2,2,2,2,2), kernel_sizes = (3,3,3,3,3,3), min_resolution = 0, max_resolution = 500):
    assert(len(strides) == len(kernel_sizes)), "strides and kernel_sizes should be in equal length!"

    padding_free_resolutions = list()

    # exhaustive search. Start with output node count of 1 to calculate the input resolution without any loss of information.
    # increase the output node count to find all the available resolution values within the specified range.
    output_node_count = 1
    while (True):
        # calculate node count in the reverse order. idx of 0 means the last layer.
        reverse_strides = strides[::-1]
        reverse_kernel_sizes = kernel_sizes[::-1]
        reverse_layer_output_node_counts = list()
        reverse_layer_input_node_counts = list()
        for idx, (stride, kernel_size) in enumerate(zip(reverse_strides, reverse_kernel_sizes)):
            if idx == 0:
                reverse_layer_output_node_counts.append(output_node_count)
            else:
                reverse_layer_output_node_counts.append(reverse_layer_input_node_counts[-1])

            # the formula calculates input node count without any information loss.
            reverse_layer_input_node_counts.append(kernel_size + stride * (reverse_layer_output_node_counts[-1] - 1))

        layer_input_node_counts = reverse_layer_input_node_counts[::-1]

        # stop if it exceeds maximum resolution
        if layer_input_node_counts[0] > max_resolution:
            break

        # store it if it is within the range of interest
        if (layer_input_node_counts[0] >= min_resolution) and (layer_input_node_counts[0] <= max_resolution):
            padding_free_resolutions.append(layer_input_node_counts[0])

        # increment output_node_count by 1
        output_node_count += 1

    return padding_free_resolutions

######################################################################################
# calculate number of parameters
######################################################################################
def count_parameters(strides, kernel_sizes, output_channel_counts, input_resolution, output_dimension):
    assert(len(strides) == len(kernel_sizes) and len(kernel_sizes) == len(output_channel_counts)), "strides, kernel_sizes, and channel_counts should be in equal length!"

    # infer input channel count from output_channel_counts
    input_channel_counts = [output_channel_count for output_channel_count in output_channel_counts[0:-1]]
    input_channel_counts.insert(0,1) # it was (0,6) for some reason and was crashing. Leaving for record.

    num_parameters = 0

    # count number of parameters in convolutional layers
    for kernel_size, input_channel_count, output_channel_count in zip(kernel_sizes, input_channel_counts, output_channel_counts):
        # For each convolutional layer, do the following.
        # add number of weights in convolution
        num_parameters += input_channel_count * kernel_size * kernel_size * output_channel_count
        # add number of biases in convolution
        num_parameters += output_channel_count

    # count number of parameters in fully connected layer
    # first, calculate the resolution of final convolutional layer
    layer_output_dimensions = list()
    for stride, kernel_size in zip(strides, kernel_sizes):
        if len(layer_output_dimensions) == 0:
            layer_input_dimension = np.asarray((input_resolution[1], input_resolution[0]))
        else:
            layer_input_dimension = layer_output_dimensions[-1]

        # calculate output_dimension based on layer_input_dimension, stride, and kernel_size
        layer_output_dimension = np.ceil((layer_input_dimension - (kernel_size - 1)) / stride)

        layer_output_dimensions.append(layer_output_dimension)

    pdb.set_trace()
    # add number of parameters in the fully connected layer to num_parameters
    num_parameters += layer_output_dimensions[-1][1] * layer_output_dimensions[-1][0] * output_channel_counts[-1] * output_dimension

    return num_parameters

######################################################################################
# checks whether the given string is a number (including non-integers)
######################################################################################
def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


######################################################################################
# generates a pandas dataframe out of ALL the events files (tensorboard log files) stored under folder_path.
######################################################################################
def parse_event_files(folder_path, tags):
    try:
        import tensorflow as tf
    except:
        print('Tensorflow and pandas are required to execute this function.\n')
        sys.exit()

    # tags should be a list, even if there is only one element.
    assert(type(tags).__name__ == 'list'), 'tags are expected to be a list.'

    # get the list of all the folders that contain event file (tensorboard log file).
    event_file_paths = glob.glob(os.path.join(folder_path,'**/*.tfevents.*'),recursive = True)

    # iterate through each folder path to generate a dictionary describing the corresponding train run.
    data = dict()
    data['descriptions'] = list()
    for tag in tags:
        data[tag] = list()

    for event_file_path in event_file_paths:
        description_str = os.path.basename(os.path.dirname(event_file_path)) # use the folder name that encloses the event file

        # '_' is the separator for variable name and values, and also between list elements.
        tokens = description_str.split('_')

        # parse the tokens. If strings appear, join them with '_'. If numbers appear, leave them separate.
        param_names = list()
        param_values = list()
        previous_token_was_string = False
        for token in tokens:
            if not is_number(token):
                # If token is a string, build up the parameter name.
                
                if not previous_token_was_string:
                    # begin a new parameter name
                    param_names.append('')
                    previous_token_was_string = True

                # accumulate tokens to build the parameter name..
                param_names[-1] = param_names[-1] + '_' + token if len(param_names[-1]) > 0 else token

            elif is_number(token):
                # If token is a number, build up the parameter value list
                if previous_token_was_string:
                    param_values.append(list())
                    previous_token_was_string = False

                # append tokens to build the parameter value list..
                param_values[-1].append(float(token))

        description = dict()
        for param_name, param_value in zip(param_names, param_values):
            description[param_name] = param_value

        data['descriptions'].append(description)

        # prepare empty lists for storing the data
        for tag in tags:
            data[tag].append(list())

        # fetch the desired information from the tensorboard log file
        for e in tf.train.summary_iterator(event_file_path):
            for v in e.summary.value:
                for tag in tags:
                    if v.tag == tag:
                        data[tag][-1].append(v.simple_value)

    return data

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

######################################################################################
# Compress all the images found in inputPath into one .zip file named outputFilename
# and put it under inputPath
######################################################################################
def compressImages(inputPath, outputFilename, removeImages = False):
    inputPath = os.path.realpath(inputPath)

    # give the appropriate extension if it was not already included in outputFilename
    if not '.zip' in outputFilename:
        outputFilename = outputFilename + '.zip'
    outputFilePath = os.path.realpath(os.path.join(inputPath, outputFilename))

    # get the list of all the image files found in the folder inputPath
    imageFiles = glob.glob(os.path.join(inputPath, '**.png'), recursive = True)

    # generate a compressed file with the given name
    print('Found %d images and compressing them into %s...'%(len(imageFiles),outputFilename))
    with zipfile.ZipFile(outputFilePath, 'w', zipfile.ZIP_DEFLATED) as zw:
        for imageFile in imageFiles:
            zw.write(imageFile, os.path.basename(imageFile))

    if removeImages:
        print('Removing the image files...')
        # remove all the image files
        for imageFile in imageFiles:
            os.remove(imageFile)

######################################################################################
# Convert .avi files to .zip, organized by label
# Subsample throws away data in the case of near duplicate images. It *does not* create a test set from the raw data.
#
# Input video files should be living in a folder named 'video' under a parent folder with an arbitrary name.
# Give the parent folder name in the list 'sourceFolders' in the beginning of the script.
# The script will automatically generate resized versions of the input images compress them into a created folder named 'compressed' under the parent folder.
######################################################################################
def convertAviToZip(inputPath, footagePath, subsample=None):

    #Only import cv2 if we call this function
    import cv2
    
    inputPath = os.path.realpath(inputPath)
    footagePath = os.path.realpath(footagePath)

    # copy csv file from input into footage path
    inputcsvFile = glob.glob(os.path.join(inputPath, "*.csv"))
    if (len(inputcsvFile) == 1):
        footagecsvfile = os.path.join(footagePath, os.path.basename(inputcsvFile[0]))
        shutil.copyfile(inputcsvFile[0], footagecsvfile)
        print('copied csv file to footage path\n')
    else:
        print('WARNING: more than one csv file in folder. Convention only allows single CSV per footage !\n')

    # local function to remove decompressed files are processing
    def remove_folder(path):
        # check if folder exists
        if os.path.exists(path):
             # remove if exists
             shutil.rmtree(path)

    decompressedFolder = inputPath + '/decompressed'

    # create footage folder if not existing
    if not os.path.exists(footagePath):
        os.mkdir(footagePath)
        
    print('    Working on folder: ' + inputPath + '...')
    
    # gather video files from input folder
    srcFiles = glob.glob(os.path.join(inputPath, "*.avi"))
    srcFiles.sort()
    numFiles = len(srcFiles)

    # process each input file    
    for progress, srcFile in enumerate(srcFiles):
    
        # create path for decompressed files if not existing
        if not os.path.exists(decompressedFolder):
            os.mkdir(decompressedFolder)

        currentImageSampleIdx = 0
        
        print('  Extracting file ' +str(progress+1) +'/' + str(numFiles) + ': ' + srcFile + '...')
        cap = cv2.VideoCapture(os.path.join(inputPath, srcFile))
        
        # create zip filename from video filename       
        zipBaseFileName = os.path.splitext(os.path.basename(srcFile))[0]
        zipFileName = os.path.join(footagePath, zipBaseFileName + '.zip')

        # perform image extraction and compression.
        idx = 0
        zipSrcFiles = list()
        while True:
    
            ret,frame = cap.read()
            # stop if next image can't be read from video
            if ret is False:
                break
            else:
                
                if (currentImageSampleIdx % subsample == 0):
                    currentImageSampleIdx = 0
                    idx += 1

                    imageFileName = format(idx,'04d') + '.jpg'
                    
                    zipSrcFiles.append(imageFileName)
                    cv2.imwrite(os.path.join(decompressedFolder, imageFileName), frame)
                
            currentImageSampleIdx += 1

        print('  Compressing: ' + zipFileName + '...')
        zf = zipfile.ZipFile(os.path.join(footagePath, zipFileName),mode = 'w')
        for zipSrcFile in zipSrcFiles:
            zf.write(os.path.join(decompressedFolder, zipSrcFile), zipSrcFile)
        zf.close()
   
        # delete decompressed folder with content
        remove_folder(decompressedFolder)

def importCSVHeader(csvfile):

    # load csv file to get header and the total number of samples
    mycsv = open(csvfile)
    csvread = csv.reader(mycsv, delimiter=',')
    csv_header = []
    rowIdx = 0
    for row in csvread:
        if (rowIdx == 0):
            csv_header = row
        rowIdx = rowIdx + 1
    mycsv.close()
    numSamples = rowIdx-1

    return (csv_header, numSamples)

def parseCSV(csvfile):

    print('parsing csv file : ', csvfile)
    (csv_header, numSamples) = importCSVHeader(csvfile)

    # create numpy arrays with specified datatypes for attributes
    labels = {}
    labellist = []
    validLabels = []
    for attrib in csv_header:
        
        # get field type
        attrib_parts = attrib.split("_")
        attribtitle = ''
        attribtitle = attrib.strip() # remove whitespaces
        
        if (len(attribtitle) == 0): # if there is no valid label skip this
            continue

        # initialize numpy array for respective label
        (dt, conversionSuccessful) = getH5Datatype(attrib_parts[0])
        labels[attribtitle] = np.empty([numSamples], dtype=dt)
                
        # save list to remember order of attribute titles
        labellist.append(attribtitle)

        if (conversionSuccessful):
            validLabels.append(attribtitle)
    
    # fill labels with values
    mycsv = open(csvfile)
    csvread = csv.reader(mycsv, delimiter=',')

    # iterate over rows in csv file
    for rowIdx, row in enumerate(csvread):
        
        # ignore header of csv file
        if (rowIdx == 0):
            continue
        
        # iterate over attributes for this row
        for attribIndex, attrib in enumerate(row):

            # get datatype
            
            # check if there is such an attribute
            if (attribIndex >= len(labellist)):
                #print("WARNING : skipping cell in csv file due to missing attribute title")
                continue

            dt = labels[labellist[attribIndex]].dtype

            # convert value to correct dataype
            if (dt == 'int'):
                value = np.asarray(attrib, np.float32)
                value = np.asarray(value, np.int32)
            else:
                value = np.asarray(attrib, dt)
 
            # add value to respective label list
            labels[labellist[attribIndex]][rowIdx-1] = value

    # close csv
    mycsv.close()

    return (labels, labellist, numSamples, validLabels)

def loadImageFromArchive(archivePath, imageFileName):
    # open archive
    img = []
    with zipfile.ZipFile(archivePath, 'r') as zipFile:
        # load image
        img = misc.imread(BytesIO(zipFile.open(imageFileName, 'r').read()))
    return img

# assumption : for each left image (L) there is an image (R) which only differs in the archive name and (string_eye)
def convertMonocularToBinocularDataset(inputFootageFolder, outputFootageFolder, referenceLabels):

    # find and parse CSV
    import glob
    csvfiles = glob.glob(os.path.join(inputFootageFolder, '*.csv'))
    assert (len(csvfiles) == 1), "Conversion function currently assumes footage folder with one single CSV file"
    csvfile =csvfiles[0]

    (labels, labellist, numSamples, validLabels) = parseCSV(csvfile)

    # create output folder
    if (os.path.exists(outputFootageFolder) == False):
        os.mkdir(outputFootageFolder)
    
    # create archive in output footage archive
    archivefile = 'archive.zip'
    archivepath = os.path.join(outputFootageFolder, archivefile)
    
    if (os.path.exists(archivepath)):
        os.remove(archivepath)
 
    # get labels required for CSV header
    eyelabel = 'string_eye'
    if eyelabel in validLabels:
        validLabels.remove(eyelabel)

    binocularLabels = []

    with zipfile.ZipFile(archivepath, mode = 'w') as zfile:

        pairedSampleIndex = 0
        for leftEyeIndex in list(range(numSamples)):

            # get new left sample 
            if (labels['string_eye'][leftEyeIndex] == 'L'):
                # search for corresponding right
                for rightEyeIndex in list(range(numSamples)):
                    if (labels['string_eye'][rightEyeIndex] == 'R'):
                        sampleMatching = True
                        for lab in referenceLabels:
                            # check if labels are matching 
                            if (labels[lab][leftEyeIndex] != labels[lab][rightEyeIndex]):
                                sampleMatching = False
                                break
                        if (sampleMatching):
                            print('matching pair', leftEyeIndex, ' - ', rightEyeIndex)

                            # load left image
                            leftImageArchive = os.path.join(inputFootageFolder, str(labels['archive'][leftEyeIndex]))
                            leftImage = loadImageFromArchive(leftImageArchive, str(labels['image'][leftEyeIndex]))
                        
                            # load right image
                            rightImageArchive = os.path.join(inputFootageFolder, str(labels['archive'][rightEyeIndex]))
                            rightImage = loadImageFromArchive(rightImageArchive, str(labels['image'][rightEyeIndex]))
                        
                            # write pair of images
                            imageNameLeft = "img_%d_L.jpg" % pairedSampleIndex
                            imageNameRight = "img_%d_R.jpg" % pairedSampleIndex
                            cv2.imwrite(imageNameLeft, leftImage)                         # write paired image to disc
                            cv2.imwrite(imageNameRight, rightImage)                         # write paired image to disc
                            zfile.write(imageNameLeft)   # add image pair to zip file
                            zfile.write(imageNameRight)   # add image pair to zip file
                            os.remove(imageNameLeft)
                            os.remove(imageNameRight)
                            
                            pairedSampleIndex = pairedSampleIndex + 1                   # increment index of paired images

                            # create combined label 
                            thisBinocularSampleLabel = 'archive,image_filename_binocular_left,image_filename_binocular_right'
                            s = ",";
                            thislabeldata = s.join((archivefile, imageNameLeft, imageNameRight))
                            for binLabel in validLabels:
                                thislabeldata = thislabeldata + ',' + str(labels[binLabel][leftEyeIndex])
                            
                            binocularLabels.append(thislabeldata)

    # create csv file in output footage folder
    binocularcsv = os.path.join(outputFootageFolder, 'footage_description.csv')
      
    if (os.path.exists(binocularcsv)):
        os.remove(binocularcsv)

    with open(binocularcsv, 'w', newline='') as csvfile:
        
        # write csv header
        binocularCsvHeader = 'archive,image_filename_binocular_left,image_filename_binocular_right'
        for lab in validLabels:
            binocularCsvHeader = binocularCsvHeader + ',' + lab
        print (binocularCsvHeader)
        csvfile.write(binocularCsvHeader + '\n')

        pairedSampleIndex = 0
        
        # iterate over binocular samples and write labels
        for sample  in binocularLabels :
            csvfile.write(sample + '\n')
            pairedSampleIndex = pairedSampleIndex + 1
        
            # print number of written frames
        print('num written binocular samples : ', pairedSampleIndex)
         
######################################################################################
# in the given directory, recursively search for any file path that contains the input string
######################################################################################
def searchPathRecursivelyForString(dir_path, input_string):
    result = False
    for file_path in glob.glob(os.path.join(dir_path,'**'),recursive = True):
        if input_string in os.path.basename(file_path):
            result = True
            break
    return result

######################################################################################
# Show numpy array image using plt. Save the image output if the filename is given.
# Optionally one can specify the value range of the output image using v_min and v_max.
# Note that values out of the range [v_min, v_max] will be clipped to 0 and 255 during while converting to the uint8 type.
# If v_min and v_max not specified, it will automatically use the minimum and maximum value in the data as v_min and v_max.
######################################################################################
def showImage(img, v_min = None, v_max = None, filename = None):
       
    import matplotlib.pyplot as plt

    if isinstance(img, torch.Tensor):
        print('showing torch tensor image')
        if (img.device.type == 'cpu'):
            print('tensor on cpu already')
            img = np.asarray(img)

        elif (img.device.type == 'cuda'):
            print('tensor copied to system memory')
            img = np.asarray(img.to("cpu", torch.float32))
        else:
            print(img.device.type )
            assert False, 'unknown tensor location'
    elif isinstance(img, np.ndarray):
        print('showing numpy image')
    else:
        print(type(img))
        assert False, 'unknown image format'

    assert (len(img.shape) < 4), "showImage() only accepts a single image (CHW or WHC)"

    # convert float to uint8 if required
    if (img.dtype == np.float32): # assumption range is [-1, 1]
        print('convert float image to uint8 image')
        v_min = -1.0 if v_min is None else v_min
        v_max = 1.0 if v_max is None else v_max
        img = (img - v_min) / (v_max - v_min) * 255.0
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

    # test if image is probably CHW instead of WHC
    if (len(img.shape) == 3):
        if (img.shape[2] > 4):
            print('convert from CHW to WHC')
            img = np.transpose(img, [1,2,0])

    if (len(img.shape) == 3):
        # if RGB image
        if (img.shape[2] == 3):
            print('prepare RGB image')

        # if 2-channel image
        if (img.shape[2] == 2):
            print('prepare 2 channel image')
            # create 2 RGB images side by
            emptyChannel = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            rgbRed = np.dstack((img[..., 0], emptyChannel, emptyChannel))
            print(rgbRed.shape)

            rgbGreen = np.dstack((emptyChannel, img[..., 1], emptyChannel))
            print(rgbGreen.shape)
            # combine to one image side-by=side
            img = np.dstack(
                (np.hstack((rgbRed[...,0], rgbGreen[...,0])),
                np.hstack((rgbRed[...,1], rgbGreen[...,1])),
                np.hstack((rgbRed[...,2], rgbGreen[...,2])))
                );
            print(img.shape)
    else:
        #if 1 channel image -> grayscale
        print('prepare grayscale image')
        img = np.dstack((img, img, img))

    print(img.shape)
    plt.figure()
    plt.imshow(img)
    plt.show()

    if filename is not None:
        plt.savefig(filename)

######################################################################################
# show PyTorch image located in GPU-memory using plt
######################################################################################
def showPyTorchTensorImage(img):
    showImage(np.asarray(img.data.cpu(), dtype=np.uint8))

######################################################################################
# customized sampling function for dataset gneration
# This function tries to distribute the test points in the attribute space as Euclidian-uniformly as possible.
# NOTE: The length of sample indices could be a little different than numSamplingValues.
# This is because the selection of sampling is based on attributes, where each attribute may have multiple samples.
######################################################################################
def generateSampleIndicesUsingLabeledSampling(numSamplingValues, numTotalSamples, attributeForSampling = None):
    # Try to import KMeans.
    try:
        from sklearn.cluster import KMeans
    except:
        # Exit if import fails.
        print("sampling based on attribute is not supported because the module 'sklearn' is missing.")
        sys.exit()

    # convert 'attributeForSampling' to a numpy array
    attributeForSampling = np.asarray(attributeForSampling)

    # expand dimension if it is a 1 dimensional array
    if len(attributeForSampling.shape) == 1:
        attributeForSampling = np.expand_dims(attributeForSampling, axis = 1)
    
    # transpose so that axis of 0 is sample dimension and axis of 1 is attribute dimension
    if attributeForSampling.shape[0] != numTotalSamples:
        attributeForSampling = np.transpose(attributeForSampling, [1,0])

    # print error message if numTotalSamples does not agree with the number of attribute vectors.
    assert(numTotalSamples == attributeForSampling.shape[0]), "numTotalSamples does not match with the attribute array size!"

    # get unique attributes along axis 0, which is by now the sample dimension
    unique_attributes = np.unique(attributeForSampling, axis = 0)

    # estimate the cluster count that will best approximate the specified numSamplingValues
    sample_count_per_unique_attribute = numTotalSamples / unique_attributes.shape[0]
    desired_cluster_count = np.round(numSamplingValues / sample_count_per_unique_attribute).astype(int)

    # use scikit k-means clustering algorithm to find the centroid values
    kmeans = KMeans(n_clusters = desired_cluster_count)
    kmeans = kmeans.fit(unique_attributes)
    centroid_attributes = kmeans.cluster_centers_

    # for each centroid, choose the samples whose attributes are close to the centroids.    
    selectedSampleIndices = np.empty(0, dtype = np.int32)
    for centroid_attribute in centroid_attributes:
        # distance from this centroid to all the unique attributes
        distances = Euclidean_dist(np.expand_dims(centroid_attribute, axis = 0), unique_attributes, axis = 1)

        # find the closest unique attribute
        closest_attribute = unique_attributes[np.argmin(distances)][:]

        # sample indices that correspond to the selected attribute
        closest_sample_indices = np.nonzero(np.prod(closest_attribute == attributeForSampling, axis = 1))[0]

        # add the selcted sample indices to the output sample indices
        selectedSampleIndices = np.concatenate((selectedSampleIndices, closest_sample_indices), axis = 0)
    return selectedSampleIndices


######################################################################################
# default uniform randomization sampling function for dataset gneration
######################################################################################
def generateSampleIndices(numSamplingValues, numTotalSamples, attributeForSampling = None):
    
    # create empty array
    selectedSampleIndices = np.empty([0],dtype=np.int32) 

    # if attributeForSampling is not empty, sample based on the passed attributes
    if attributeForSampling is not None:
       return generateSampleIndicesUsingLabeledSampling(numSamplingValues, numTotalSamples, attributeForSampling)

    # else randomize samples
    while len(selectedSampleIndices) < numSamplingValues:
        newIndex = int( math.floor(random.random() * numTotalSamples))
        selectedSampleIndices = np.append(selectedSampleIndices, [newIndex])
        selectedSampleIndices = np.unique(selectedSampleIndices)

    return selectedSampleIndices

# function assumes CHW
def convertImageToGrayscale(img):
    
    sourceDataType = img.dtype;

    # convert to float32
    imgFloat = np.asarray(img, dtype=np.float32);

    imgGrayScale = (imgFloat[0] + imgFloat[1] + imgFloat[2]) / 3.0;
    return np.asarray(imgGrayScale, dtype=sourceDataType);

# function assumes CHW
def convertGrayscaleToRGB(img):

    return np.vstack((img,img,img));

def writeCSVFile(labels, outputfile, limitNumberOfSamples = None):

    if (os.path.exists(outputfile)):
        os.remove(outputfile)
    
    with open(outputfile, 'w', newline='') as csvfile:
        
        # write csv header
        header = None
        
        for key, value in labels.items():
            if (header is None):
                header = key
            else:
                header = header + ',' + key

        print (header)
        csvfile.write(header + '\n')

        thekeys = list(labels.keys())
        numSamples = len(labels[thekeys[0]])
        
        if (limitNumberOfSamples is not None):
            if (limitNumberOfSamples < numSamples):
                numSamples = limitNumberOfSamples


        # write labels
        for sampleIndex in range(numSamples):
            for keyIdx, key in enumerate(thekeys):
                csvfile.write(str(labels[key][sampleIndex]))
                if (keyIdx < len(thekeys) - 1):
                    csvfile.write(',')
            csvfile.write('\n')
        
    # print number of written frames
    print('num written samples : ', numSamples)
    print('written csv file : ', outputfile)

# function merging csvfiles by finding the intersection of labels
def createCombinedFootageFile(csvfilelist, outputfile):

    print("Create combined csvfile")

    all_labels = []
    all_labellist = []
    all_numSamples = []
    all_validLabels = []

    collectedLabels = []

    # collect all available labels
    for csvfile in csvfilelist:
        labels, labellist, numSamples, validLabels = parseCSV(csvfile)

        for l in labellist:
            collectedLabels.append(l)

    print('collectedLabels : ', collectedLabels)

    uniqueLabels = set(collectedLabels)
    print('uniqueLabels : ', uniqueLabels)

    uniqueLabelsList = list(uniqueLabels)

    # remove labels if not contained in all files
    for csvfile in csvfilelist:
        labels, labellist, numSamples, validLabels = parseCSV(csvfile)
        
        # remove labels which are not contained in this specific dataset
        toBeRemoved = []
        for l in uniqueLabels:
            if l not in labellist:
                toBeRemoved.append(l)
        uniqueLabels -= set(toBeRemoved)

    print('uniqueLabels after intersection : ', uniqueLabels)
    
    # VALIDITY CHECK (based on available image list and label list)
    
    merged_validLabels = []     # check if there is a least one valid label left
    merged_images = []          # check if there is at least on valid image description left

    for l in uniqueLabels:
        
        parts = str.split(l, '_')
                
        if (len(parts) > 0):
            # check for valid label
            dt, conversionSuccessful = getH5Datatype(parts[0])
            if (conversionSuccessful):
                merged_validLabels.append(l)
            # check for image
            if (parts[0] == 'image'):
                merged_images.append(l)

    csvfile_valid = True
    if (len(merged_validLabels) == 0):
        csvfile_valid = False
        print("ERROR: Merging csv file results in invalid footage since there is no common valid label !!!")
    if (len(merged_images) == 0):
        csvfile_valid = False
        print("ERROR: Merging csv file results in invalid footage since there is no image description !!!")

    if (csvfile_valid):

        # WRITE MERGED FOOTAGE FILE

        # dictionary with all labels    
        combinedLabels = {}

        # combine samples for archive, images, valid labels
        for csvfile in csvfilelist:
            labels, labellist, numSamples, validLabels = parseCSV(csvfile)

            # iterate over each label in dictionary
            for key, lab in labels.items():
                print(key)
                if key in uniqueLabels:
                        
                    # change archive file path to absolute paths
                    if (key == 'archive'):
                        basepath = os.path.dirname(csvfile)
                        datatype = lab[0].dtype
                        for idx in range(len(lab)):
                            #pdb.set_trace()
                            lab[idx] = np.array(os.path.join(basepath, str(lab[idx])), datatype)
                    
                    if (key not in combinedLabels):
                        combinedLabels[key] = lab
                    else:
                        combinedLabels[key] = np.concatenate((combinedLabels[key],lab))

        # write csv
        writeCSVFile(combinedLabels, outputfile)

    return outputfile, csvfile_valid
     

######################################################################################
# end of utils.py
######################################################################################
