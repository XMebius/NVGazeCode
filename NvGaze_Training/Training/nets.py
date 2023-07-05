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

from __future__ import print_function
import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image
import os, platform
import pdb, sys
from copy import deepcopy

def kaiming_weights_init(m):
    classname = m.__class__.__name__
    gain = nn.init.calculate_gain('relu')
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight, gain)

# A lean net whose number of layer, stride, and number of features are configurable.
class ConfigurableMultiViewLeanNetNoPadding(nn.Module):
    def __init__(self, dropout = 0.1, fc_node_counts = (2,), input_resolution = (255, 191), strides = (2,2,2,2,2,2), kernel_sizes = (3,3,3,3,3,3), output_channel_counts = (16, 24, 36, 54, 81, 122), numEVC = 6):
        super().__init__() #marker

        assert(len(strides) == len(output_channel_counts)), "Number of stride values has to match number of output channel values"
        assert(len(strides) == len(kernel_sizes)), "Number of stride values has to match number of kernel values"

        # input channels per layer derived from output channels
        input_channel_counts = list(output_channel_counts[0:-1])
        input_channel_counts.insert(0,numEVC) # add ExVxC input as first input
        input_channel_counts = tuple(input_channel_counts) # convert to tuple

        self.conv = list()  # init empty list for convolutional layers
        self.drop = list()  # init empty list for drop out layers
        layer_output_dimensions = list() # init empty list for output dimensions

        # iterate over stride, kernel size, input_channels, output_channels with corresponding index
        for stride, kernel_size, input_channel_count, output_channel_count in zip(strides, kernel_sizes, input_channel_counts, output_channel_counts):

            # create and append convolution layer
            self.conv.append(nn.Conv2d(input_channel_count, output_channel_count, kernel_size, stride))
            # create and append drop out layer
            self.drop.append(nn.Dropout2d(p = dropout))

            # compute output dimensions
            if len(layer_output_dimensions) == 0: # if there are no output dimension yet, input is image dimensions
                input_dimension = np.asarray((input_resolution[1], input_resolution[0]))
            else:
                input_dimension = layer_output_dimensions[-1] # otherwise input dimension is size of output dimension of previous layer

            # calculate output_dimension based on input_dimension, stride, and kernel_size
            # padding is assumed to be zero at this point !
            output_dimension = np.ceil((input_dimension - (kernel_size - 1)) / stride)

            layer_output_dimensions.append(output_dimension)

        layer_output_dimensions = np.asarray(layer_output_dimensions)
        # compute node count of last layer as product of output dimensions times number of output channels of last convolutional layer
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv[-1].out_channels) # times numviews due to multiview case
        
        # create fully-connected layers
        self.fc = list()
        for idx, output_node_count in enumerate(fc_node_counts):
            if idx == 0:
                input_node_count = self.last_layer_node_count # size of first fc layer
            else:
                input_node_count = fc_node_counts[idx - 1] # for second and following fc layers : input nodes of fc layer equals output nodes of previous fc layer
            self.fc.append(nn.Linear(input_node_count, output_node_count)) # create and append fully-connected layer

        # convert the lists of conv, dropout, and fully-connected layers into nn.ModuleList's.
        self.conv = nn.ModuleList(self.conv)
        self.drop = nn.ModuleList(self.drop)
        self.fc   = nn.ModuleList(self.fc)

    # describe forward pass
    # arguments :
    #   x : input data (sample)
    def forward(self,x):
        i = 1
        # iterate and apply pairs convolutional layers and drop out layers
        for conv, drop in zip(self.conv, self.drop):
            # RELU
            x = F.relu(conv(x))
            # INSTANCE NORM
            if i < 5:
                m = nn.InstanceNorm2d(x.size()[1],eps=1e-06, momentum=0.1, affine=False, track_running_stats=False)
                x = m(x)
            i += 1
            x = drop(x)

        # get shape after convolution layers
        x = x.view(-1, self.last_layer_node_count) # reshape output to fit (first) fully-connected layer
        # iterate and apply fully-connected layers
        for fc in self.fc:
            x = fc(x)
        
        # return result
        return x

    # describe cuda usage
    def cuda(self):
        super().cuda()
        
        for conv, drop in zip(self.conv, self.drop):
            conv.cuda()
            drop.cuda()

        for fc in self.fc:
            fc.cuda()
    
     
    def genFeatureMaps(self,x,featureSpacing):
        featureMaps = list()
        featureDimensions = list()

        for conv in self.conv:
            x = F.relu(conv(x)) # apply convolution and reLU
            featureMaps.append(torchvision.utils.make_grid(x.data.squeeze().unsqueeze(1), nrow = 4, padding = featureSpacing).cpu().numpy()[0])
            featureDimensions.append((x.data.size()[2],x.data.size()[3])) # dimensions following K C H W 

        return featureMaps, featureDimensions

class CalibAffine2(nn.Module):
    '''A calibration network of 2x2 affine transform for multi-user calibration'''
    def __init__(self, subjects):
        super().__init__()
        self.user = nn.ModuleList()
        for i in range(subjects):
            fc = nn.Linear(2,2)
            torch.nn.init.eye(fc.weight)
            torch.nn.init.constant(fc.bias,0)
            self.user.append(fc)
    def forward(self,x,subjectID):
        x = self.user[int(subjectID)](x)
        return x

class CalibAffine8(nn.Module):
    '''A calibration network of 2x2 affine transform for multi-user calibration'''
    def __init__(self, subjects):
        super().__init__()
        self.user = nn.ModuleList()
        for i in range(subjects):
            fc = nn.Linear(8,2)
            torch.nn.init.eye(fc.weight)
            torch.nn.init.constant(fc.bias,0)
            self.user.append(fc)
    def forward(self,x,subjectID):
        x = self.user[int(subjectID)](x)
        return x

class CalibNet(nn.Module):
    def __init__(self, network, calibration):
        super().__init__()
        self.add_module('net',network)
        self.add_module('calib',calibration)
    def forward(self,x,subjectID):
        x = self.net(x)
        subjects = np.unique(subjectID)
        output = Variable(torch.zeros(x.size(0),2)).cuda()
        for subject in subjects:
            mask = Variable(torch.nonzero(subjectID == int(subject))[:,0]).cuda()
            subset = torch.index_select(x,0,mask)
            outset = self.calib(subset,subject)
            output[mask] = outset
        return output


    def __init__(self, dropout = 0.1, num_output_nodes = 2, input_resolution = (320, 240)):
        super().__init__() #marker
        
        self.conv1Left = nn.Conv2d(1,16,3,stride=2)
        self.drop1Left = nn.Dropout2d(p=dropout)
        self.conv1Right = nn.Conv2d(1,16,3,stride=2)
        self.drop1Right = nn.Dropout2d(p=dropout)
        
        self.conv2Left = nn.Conv2d(16,24,3,stride=2)
        self.drop2Left = nn.Dropout2d(p=dropout)
        self.conv2Right = nn.Conv2d(16,24,3,stride=2)
        self.drop2Right = nn.Dropout2d(p=dropout)

        self.conv3Left = nn.Conv2d(24,36,3,stride=2)
        self.drop3Left = nn.Dropout2d(p=dropout)
        self.conv3Right = nn.Conv2d(24,36,3,stride=2)
        self.drop3Right = nn.Dropout2d(p=dropout)

        self.conv4Left = nn.Conv2d(36,54,3,stride=2)
        self.drop4Left = nn.Dropout2d(p=dropout)
        self.conv4Right = nn.Conv2d(36,54,3,stride=2)
        self.drop4Right = nn.Dropout2d(p=dropout)

        self.conv5Left = nn.Conv2d(54,81,3,stride=2)
        self.drop5Left = nn.Dropout2d(p=dropout)
        self.conv5Right = nn.Conv2d(54,81,3,stride=2)
        self.drop5Right = nn.Dropout2d(p=dropout)
        
        self.conv6Left = nn.Conv2d(81,122,3,stride=2)
        self.drop6Left = nn.Dropout2d(p=dropout)
        self.conv6Right = nn.Conv2d(81,122,3,stride=2)
        self.drop6Right = nn.Dropout2d(p=dropout)

        # calculate output dimensions
        conv_layer_count = 6
        layer_output_dimensions = np.zeros((conv_layer_count,2))
        for layer_index in range(conv_layer_count):
            if layer_index == 0:
                input_dimension = (input_resolution[1], input_resolution[0])
            else:
                input_dimension = layer_output_dimensions[layer_index - 1]
            # effect of stride of 2
            layer_output_dimensions[layer_index,:] = np.ceil(np.asarray(input_dimension) / 2) - 1
        # Simply multiplying by two is correct here because the convolution path is fully duplicated even though some kernel weights are shared
        self.last_layer_node_count = int(np.prod(layer_output_dimensions[-1,:]) * self.conv6Left.out_channels) * 2
        self.fc1 = nn.Linear(self.last_layer_node_count,num_output_nodes) # 976 inputs (122 layers, resolution of 4 x 2)

    def forward(self,x):
        #Split into left and right eye images.
        left = x[:,0:1,:,:]
        right = x[:,1:2,:,:]
        #Conv1
        left = F.relu(self.conv1Left(left))
        left = self.drop1Left(left)
        right = F.relu(self.conv1Right(right))
        right = self.drop1Right(right)
        #Conv2
        left = F.relu(self.conv2Left(left))
        left = self.drop2Left(left)
        right = F.relu(self.conv2Right(right))
        right = self.drop2Right(right)
        #Switch here to separate convolution paths
        #Conv3
        left = F.relu(self.conv3Left(left))
        left = self.drop3Left(left)
        right = F.relu(self.conv3Right(right))
        right = self.drop3Right(right)
        #Conv4
        left = F.relu(self.conv4Left(left))
        left = self.drop4Left(left)
        right = F.relu(self.conv4Right(right))
        right = self.drop4Right(right)
        #Conv5
        left = F.relu(self.conv5Left(left))
        left = self.drop5Left(left)
        right = F.relu(self.conv5Right(right))
        right = self.drop5Right(right)
        #Conv6
        left = F.relu(self.conv6Left(left))
        left = self.drop6Left(left)
        right = F.relu(self.conv6Right(right))
        right = self.drop6Right(right)
        x = torch.cat((left, right), 1)
        x = x.view(-1, self.last_layer_node_count)
        x = self.fc1(x)
        return x