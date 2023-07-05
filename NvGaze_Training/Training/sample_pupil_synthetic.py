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

'''
NOTE: all modules imported must have '_' underscore prefix
'''

import os as _os
import platform as _platform # renaming of the model is necessary for the serializer ignore the modules (which can't be pickled).


# Base directory
data_dir = 'C:/NvGaze/Datasets/h5/'

# Dataset configuration.

dataset = 'combinedSyntheticDataset_293.h5'

train_data = []
test_data = []

traindatasetfile = dataset
traindatasetfile.replace('.h5', '_train.h5')
train_data.append(traindatasetfile)

testdatasetfile = dataset
testdatasetfile.replace('.h5', '_test.h5')
test_data.append(testdatasetfile)


torchPresent = False
try:
    import nets as _nets
    import torch.nn as _nn
    torchPresent = True
except ImportError:
    pass
import logging as _logging
import sys as _sys

if torchPresent:
    network_type            = _nets.ConfigurableMultiViewLeanNetNoPadding    # a class of network
    # When loading pre-trained network, set network_init to _nets.load_pretrained and provide the path to the weight in the 'weight_path' variable.
    network_init            = _nets.kaiming_weights_init            # method to initalize the weights
    output_description      = ['float_pupil_x', 'float_pupil_y']

    input_resolution        = (293, 293)

    loss_func               = _nn.MSELoss

datasetloader               = 'DatasetPupilcenter.py'
usedDataPercentage          = 100.0

num_epochs                  = 200                                   # Number of epochs to train.
rampup_length               = 0
rampdown_length             = 50
start_epoch                 = 0                                     # Starting epoch.
batch_size                  = 20                                    # Samples per minibatch.
learning_rate               = 0.0001                                # Multiplier for updating weights every batch
momentum                    = 0.9                                   # Training momentum to prevent getting stuck in local minima
adam_beta1                  = 0.9                                   # Default value.
adam_beta2                  = 0.999                                 # Tends to be more robust than 0.999.
adam_epsilon                = 1e-8                                  # Default value.
dropout                     = 0.1                                   # Proportion dropout
checkpoint_frequency        = 10                                    # Frequency of weight dumping during training.

log_level                   = _logging.INFO                         #

# 5-layer network for 127x63
fc_node_counts              = [2,]

## configure with decreased kernel sizes
strides                     = [2,2,2,2,2,2,2]
kernel_sizes                = [9,7,5,5,3,3,3]
output_channel_counts       = [24, 36, 52, 80, 124, 256, 512]
numEVC                      = 1 