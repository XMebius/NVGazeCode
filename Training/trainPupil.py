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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging, sys, shutil, time, argparse, os, importlib.util
from PIL import Image
import platform as _platform
import h5py
import numpy as np
import math
import time

# import utils
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

datasetRootPath = 'C:/NvGaze/Datasets/h5/'

class RegressLoss(torch.nn.Module):
    
    def __init__(self):
        super(RegressLoss,self).__init__()
        
    def forward(self,x,y):
        totloss = torch.sum((x[:,0]-y[:,0])**2 + (x[:,1]-y[:,1])**2)
        return totloss


#Tensorboard
spec = importlib.util.spec_from_file_location('TensorBoardLogger', os.path.join(os.path.dirname(os.path.realpath(__file__)),'TensorBoardLogger.py'))
TensorBoardLogger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TensorBoardLogger)


def set_optimal_affine_xform(raw_values, targets):
    assert raw_values.shape == targets.shape and targets.shape[0] >= 3
    if type(raw_values).__name__ == 'Tensor':
        raw_values = raw_values.cpu().detach().numpy()
    if type(targets).__name__ == 'Tensor':
        targets = targets.cpu().detach().numpy()
    P = np.concatenate((raw_values, np.ones((raw_values.shape[0],1))), axis = 1)
    Q = np.concatenate((targets, np.ones((targets.shape[0],1))), axis = 1)
    M = np.dot(np.dot(Q.T, P), np.linalg.inv(np.dot(P.T, P)))
    return np.transpose(M, (1,0)) # multiply this to the raw_values to get the optimally Affine-transformed to match with targets

def adjust_learning_rate(optimizer, config, epoch):
    """Ramp up and ramp down of learning rate"""
    if epoch < config.rampup_length:
        p = max(0.0, float(epoch)) / float(config.rampup_length)
        p = 1.0 - p
        rampup_value = math.exp(-p*p*5.0)
    else:
        rampup_value = 1.0
    if epoch >= (config.num_epochs - config.rampdown_length):
        ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
        rampdown_value = math.exp(-(ep * ep) / config.rampdown_length)
    else:
        rampdown_value = 1.0
    adjusted_learning_rate = rampup_value * rampdown_value * config.learning_rate
    logging.debug('   Learning rate: %s '%adjusted_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adjusted_learning_rate

def save_checkpoint(state, isBest, checkFile, bestFile, net):
    torch.save(state, checkFile)
    if isBest:
        shutil.copyfile(checkFile, bestFile)
        # save complete model    
        torch.save(net, checkFile.replace(".pth", "_model.pth"))

def runTrainingAlternating(net, train, test, lossFunc, optimizer, config, eval_sets = None, **kwargs):

    print('runTrainingAlternating')

    # start new training or continue training (finetuning)
    if config.network_init.__name__ != 'load_pretrained': # New training. Start fresh.
        bestLoss = float('inf')
        allLosses = []
        allEpochs = [[],{}]
        first_epoch = 0
    else:
        bestLoss = kwargs['bestLoss']
        allLosses = kwargs['allLosses']
        allEpochs = kwargs['allEpochs']
        first_epoch = kwargs['first_epoch']

    startTime = time.time()
    last_epoch = first_epoch + config.num_epochs        

    # initialize tensorboard logger
    loggerTF = TensorBoardLogger.TensorBoardLogger(True, "", os.path.join(kwargs['dump_path'], kwargs['run_description']))

    for epoch in range(first_epoch, last_epoch):

        # adjust learning rate
        adjust_learning_rate(optimizer, config, epoch - first_epoch)

        # initialize parameters for training
        isBest = False
        runningLoss = 0.0
        epochStart = time.time()
        net.train() # activate dropouts

        logging.debug('   Shuffling the data...')
        # shuffle each training dataset
        for individual_train_set in train:
            individual_train_set.shuffle() # shuffle the data set indices
        logging.debug('   Shuffling ended. Starting training loop.')

        # For the case of multiple data loaders, one batch set has samples as many as [data loader count] x [batch_size]
        num_samples = 0

        for idx, individual_train_set in enumerate(train):

            # find out how many batches we need to run in an epoch
            num_batches = len(train[idx])
            # don't run the last batch if the sample in it is lower than batch size.
            if (len(train[idx]) - (num_batches-1)*config.batch_size) < config.batch_size:
                num_batches -= 1
                            
            for ii in range(num_batches):
            
                # get the inputs
                inputLoadingStartTime = time.time()
                inputs, region_maps, labels = individual_train_set[ii]
                inputLoadingDuration = time.time() - inputLoadingStartTime

                gradStartTime = time.time()
                # clone and turn on requires_grad
                inputs.requires_grad_()
                # zero the parameter gradients
                optimizer.zero_grad()
                gradDuration = time.time() - gradStartTime

                # forward + backward + optimize
                optimizationStartTime = time.time()
                outputs = net(inputs)

                trainLoss = lossFunc(outputs, labels) # MSE
                trainLoss.backward()

                optimizationDuration = time.time() - optimizationStartTime

                lossStartTime = time.time()
                optimizer.step()
                runningLoss += trainLoss.detach().cpu().numpy()

                allLosses.append(trainLoss.item())
                lossDuration = time.time() - lossStartTime
            
                num_samples += inputs.shape[0]

        runningLoss = runningLoss / num_samples
        allEpochs[0].append(runningLoss)


        # initialize parameters for testing
        net.eval() # disable drop-outs
        testLoss = 0.0

        # find out how many batches we need to run in an epoch
        num_batches = len(test[0])

        num_samples = 0
        # For the case of multiple data loaders, one batch set has samples as many as [data loader count] x [batch_size]
        for idx, individual_test_set in enumerate(test):
            
            num_batches = len(test[idx])
            
            for ii in range(num_batches):

                # get the inputs
                inputs, region_maps, labels = individual_test_set[ii]

                # we don't need backward propagation. disable requires_grad
                inputs.requires_grad = False
                outputs = net(inputs)
                testLoss += lossFunc(outputs, labels).detach().cpu().numpy()

                num_samples += inputs.shape[0]

        # normalize test loss over all samples
        testLoss = testLoss/num_samples
        
        if 'test' not in allEpochs[1]:
            allEpochs[1]['test'] = []
        allEpochs[1]['test'].append(testLoss)

        # print statistics
        if testLoss < bestLoss:
            bestLoss = testLoss
            isBest = True

        # save the current weights - every epoch?
        if (epoch-first_epoch+1) % config.checkpoint_frequency == 0 or isBest or (epoch-first_epoch+1)==config.num_epochs:
            calib = {}
            save_checkpoint({ # calibration not implemented yet.
                'epoch': epoch + 1,
                'all_loss': allLosses,
                'all_epochs': allEpochs,
                'config': {k: config.__dict__[k] for k in [i for i in dir(config) if i[:1] != '_']},
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss': bestLoss,
                'calibration': calib,
            },isBest,os.path.join(kwargs['dump_path'], kwargs['run_description'] + '_weights_latest.pth'), os.path.join(kwargs['dump_path'], kwargs['run_description'] + '_weights_best.pth'), net)
        
        trainingLossRMSE = np.sqrt(runningLoss)
        testLossRMSE = np.sqrt(testLoss)

        # print epoch summary
        logging.info('  Training Epoch %d:   Running Time- %.2f    Training Loss (RMSE)- %.4f    Test Loss (RMSE)- %.4f' % (epoch + 1, time.time()-epochStart, trainingLossRMSE, testLossRMSE))
        print('  Training Epoch %d:   Running Time- %.2f    Training Loss (RMSE)- %.4f    Test Loss (RMSE)- %.4f' % (epoch + 1, time.time()-epochStart, trainingLossRMSE, testLossRMSE))
        summary_loss_dict = {"train_loss": trainingLossRMSE, "test_loss": testLossRMSE}
        individual_loss_dict = dict()

        loggerTF.log_dict(summary_loss_dict)
        loggerTF.log_dict(individual_loss_dict)
    
    # print final summary
    logging.info('  Training total running time: %.2f seconds'%(time.time()-startTime))

def main(config, dump_path = None, run_description = None):
    
    # Check whether dump_path exists and create one if not existing.
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    # default/custom dataset loader
    loaderfile = 'datasets.py'
    if (config.datasetloader is not None):
        loaderfile = config.datasetloader
        logging.info('  using custom dataset loader' + config.datasetloader)

    # load dataset loader module
    spec = importlib.util.spec_from_file_location('datasets', os.path.join(os.path.dirname(os.path.realpath(__file__)), loaderfile))
    datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets)

    # Turn on logging.
    logging.basicConfig(filename=os.path.join(args.dump_path, args.run_description + '_log.txt'), level=config.log_level)

    global datasetRootPath
    
    logging.info('  Loading training data...')
    # List of train data paths
    train_data_paths = [os.path.join(datasetRootPath, dataset) for dataset in config.train_data]

    # train is a list of data loader for training data
    train = list()

    for idx, train_data_path in enumerate(train_data_paths):

        dataPerc = 100.0
        if (config.usedDataPercentage is not None):
            dataPerc = config.usedDataPercentage
            print('  using data percentage ' + str(dataPerc))
            logging.info('  using data percentage ' + str(dataPerc))

        d = datasets.Dataset(train_data_path, config.output_description, config.batch_size, input_resolution = config.input_resolution, augmentation=True, usedDataPercentage=dataPerc)

        train.append(d)
        logging.info('  Finished loading %d out of %d train data sets.'%(idx + 1, len(train_data_paths)))
        logging.info('     %d images loaded.'%(train[-1].images_np.shape[0]))

    logging.info('  Loading test data...')
    test_data_paths = [os.path.join(datasetRootPath, dataset) for dataset in config.test_data]

    # test is a list of data loader for test data
    test = list()

    for idx, test_data_path in enumerate(test_data_paths):

        dataPerc = 100.0
        if (config.usedDataPercentage is not None):
            dataPerc = config.usedDataPercentage
            print('  using data percentage' + str(dataPerc))
            logging.info('  using data percentage' + str(dataPerc))

        d = datasets.Dataset(test_data_path, config.output_description, config.batch_size, input_resolution = config.input_resolution, augmentation=False, usedDataPercentage=dataPerc)

        test.append(d)
        logging.info('  Finished loading %d out of %d test data sets.'%(idx + 1, len(test_data_paths)))
        logging.info('     %d images loaded.'%(test[-1].images_np.shape[0]))

    logging.info('  Data loading complete.  Initializing network.')
    
    print('  Data loading complete.  Initializing network.')

    resolution = config.input_resolution
    resolution = (293,293)


    net = config.network_type(dropout=config.dropout,
                              fc_node_counts = config.fc_node_counts,
                              input_resolution = resolution,
                              strides = config.strides,
                              kernel_sizes = config.kernel_sizes,
                              output_channel_counts = config.output_channel_counts,
                              numEVC = config.numEVC)

    logging.info('  Network built.  Initializing weights.')
    if config.network_type.__name__ == 'MultiSubjectsNet': # TODO: This should somehow go into configuration
        # initialize weights of core network according to the specified method.
        net.core.apply(config.network_init)
        # initialize the calibration layer (Affine transformation layer) with identity matrix.
        for calib in net.calib:
            torch.nn.init.eye_(calib.weight)
            calib.weight.requires_grad_() # this shouldn't be necessary in the future. (bug in 0.4.0, https://discuss.pytorch.org/t/since-0-4-using-nn-init-eye-disables-gradient/19999)
    else:
        net.apply(config.network_init)
    
    print('  Weights initialized.')
    logging.info('  Weights initialized.')

    print('  Transferring to graphics card.')
    logging.info('  Transferring to graphics card.')
    net.cuda()

    print('  Network initialized.')
    logging.info('  Network initialized.')
    # setup loss function
    
    print('  Setting up loss and optimizer.')
    logging.info('  Setting up loss and optimizer.')
    lossFunc = config.loss_func()
    lossFunc = RegressLoss()

    if config.network_type.__name__ == 'ResNet50':
        params_to_train = net.fc
    else:
        params_to_train = net
    
    # Adam optimizer
    optimizer = optim.Adam(params_to_train.parameters(), 
                            lr=config.learning_rate, 
                            betas=[config.adam_beta1,config.adam_beta2], 
                            eps=config.adam_epsilon, 
                            weight_decay=(config.weight_decay if hasattr(config, 'weight_decay') else 0)
    )

    # train network
    print('  Starting training.')
    logging.info('  Starting training.')
    runTrainingAlternating(**locals())

def loadModule(file, name='config'):
    if sys.version_info[0] == 3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info[0] == 2:
        import imp
        module = imp.load_source(name,file)
    return module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default = 'config.py', help = 'Path to the config file to be used. Default is config.py in the same directory as the current train.py.')
    parser.add_argument('-d', '--dump_path', default = 'dump', help = "Path to where tensorboard log, loss log, and weight files will be stored. If not provided, they will be stored in the 'dump' folder where this train.py lives.")
    parser.add_argument('-r', '--run_description', default = 'test_run', help = 'Description of run. This is used for naming train log, tensorboard log, and locating the location of the copied scripts.')
    args = parser.parse_args()
    config = loadModule(args.config_path)

    main(config, args.dump_path, args.run_description)

