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
import h5py

import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../.")

if __name__ == "__main__":

    videofiles = [
        "C:/NvGaze/Datasets/raw/video/sampleVideo.avi"
        ]

    dataset_output_path = 'C:/NvGaze/Datasets/'
    dst_setname = "sampleDataset.h5"

    sampleReductionFactor   = 10 # 1 - take all sample, 10 - use every 10th sample....

    convert_to_grayscale    = True
    eyes                    = 'monocular'

    flipImgeHorizontal      = True

    eyeSelection            = 'L' # L or R
    #eyeSelection            = 'R' # L or R
    
    output_resolution       = [640,480] # images will be resized to this resolution

    # create output path
    dst_setname = os.path.join(dataset_output_path, dst_setname)
    print("write h5 file : ", dst_setname)

    # write h5 file
    with h5py.File(dst_setname, 'w') as h5_file:

        d = h5_file.create_dataset('images',
                                    (1000, 1, 1, 1, output_resolution[1], output_resolution[0]),
                                    maxshape=(None, 1, 1, 1, output_resolution[1], output_resolution[0]),
                                    dtype=np.uint8,
                                    chunks=True)

        totalFrames = 0

        for f in videofiles:
            
            print(f)

            # open video
            cap = cv2.VideoCapture(f)
            videoFramesInMemory = []
            numFramesInCurrentVideo = 0

            usedFramesInVideo = 0

            while True:

                try:
                    ret,frame = cap.read()
                    # stop if next image can't be read from video
                    if ret is False:
                        break
                except:
                    print("Unexpected error while reading frame")
                    continue
                    

                if (numFramesInCurrentVideo % sampleReductionFactor == 0):

                    if (frame.shape[0] > 4):
                        frame = np.transpose(frame, [2, 0, 1])

                    #pdb.set_trace()

                    # convert to grayscale
                    if (convert_to_grayscale):
                        type = frame.dtype
                        frame_float = np.sum(np.asarray(frame, np.float), 0, np.float, None, np.True_) / float(frame.shape[0])
                        frame = np.asarray(frame_float, type)

                    # resize frame if needed
                    frame = resizeImage(frame, output_resolution)

                    #pdb.set_trace()

                    if flipImgeHorizontal:
                        frame = np.flip(frame,2)

                    frame = np.expand_dims(np.expand_dims(frame,0),0)

                    # add to list of frames
                    videoFramesInMemory.append(frame)

                    # iterate used frames
                    usedFramesInVideo += 1
                    print (usedFramesInVideo)

                # iterate frames in video
                numFramesInCurrentVideo += 1                        

            print ('%d frames used from all %d frames in %s'%(usedFramesInVideo,numFramesInCurrentVideo,f))
                               
            #print('resizing h5 file')
            d.resize((totalFrames+usedFramesInVideo, 1, 1, 1, output_resolution[1], output_resolution[0]))
                
            #print('writing content into h5 file')

            # write to h5 file
            for idx in range(usedFramesInVideo):
                d[totalFrames + idx, ...] = videoFramesInMemory[idx]

            totalFrames += usedFramesInVideo
            print ('%d total frames written\n'%(totalFrames))
        
    print('\n\n all done!')
