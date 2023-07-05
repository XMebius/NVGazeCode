
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

import os, sys, getpass, argparse, logging, shutil, time, itertools, copy, platform, pdb, glob, zipfile
import numpy as np
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

def trainingOutputFolder():
    outPath = 'C:/NvGaze/'
    return outPath

def configDiff(orig, config):
    # not going to look for removed items - that is not the point
    diff = {}
    for k,v in config.__dict__.items():
        if not hasattr(orig, k) or getattr(orig,k) != v:
            diff[k] = v
    return diff

def copyModule(old):
    new = type(old)(old.__name__, old.__doc__)
    new.__dict__.update(old.__dict__)
    for k,v in new.__dict__.items():
        if isinstance(v, dict):
            new.__dict__[k] = copy.copy(v)
    return new

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


def copy_scripts(orig_config_path, config_diff, code_dir_path):
    # Copy scripts from the training directory to the target result directory.
    if not (os.path.exists(code_dir_path)): os.makedirs(code_dir_path)
    scripts_to_copy = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), "*.py"))

    logging.info('Copying python code to %s'%(code_dir_path))
    for s in scripts_to_copy:
        shutil.copy(s, code_dir_path)

    # Copy the utils file as well
    shutil.copy(os.path.join('utils.py'), code_dir_path)

    # copy config file.
    logging.info('Copying config from %s to %s'%(orig_config_path,code_dir_path))
    shutil.copy(orig_config_path, code_dir_path)
    shutil.move(os.path.join(code_dir_path, os.path.basename(orig_config_path)), os.path.join(code_dir_path, 'config.py')) # Rename the config file as 'config.py' in code_dir_path. This is because we will run trainPupil.py with default option, which looks for 'config.py' in the same directory as itself. Note that this will overwrite the existing config.py.
    # Append changes at the end of the copied config.py
    with open(os.path.join(code_dir_path,'config.py'), 'a') as f:
        f.write('\n') # Append to an existing string will result in an error if the last line is not empty. Avoid it by appending a new line character.
        for k, v in config_diff.items():
            if k == 'train_data' or k == 'test_data' or k == 'eval_data':
                f.write("%s = [%s]\n"%(k,v)) # Data path should be a list of strings.
            else:
                f.write('%s = %s\n'%(k,v))

    # Generate a zip file of the scripts. The code directory under the target result directory will be erased after the run finishes.
    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED
    zf = zipfile.ZipFile(code_dir_path + '.zip', mode = 'w')
    scripts_to_copy = glob.glob(os.path.join(code_dir_path, "*.py"))
    for s in scripts_to_copy:
        zf.write(s, arcname = os.path.basename(s), compress_type = compression)
    zf.close()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default = 'local', help="'local' runs training locally (this is the default). 'remote' submits training job(s) to a cluster.")
    parser.add_argument('-v', '--var', nargs='*', action='append', help='A variable and value pair or variable and range for multiple submits')
    parser.add_argument('-e', '--experiment_name', default = 'test', help = 'Describe the experiment. If not given, it will assume you are testing your code and will put the dump files under [your network drive]/results/test/')
    parser.add_argument('-l', '--experiment_detail', default = '', help = 'Experiment specifier for run description')
    parser.add_argument('-c', '--config_path', default='config.py', help='Path to the config file to be used.')
    parser.add_argument('-j', '--job_name', default= None, help='The name of the job to be submitted.')
    args = parser.parse_args()

    # Load the config file if the given config path exists.
    if os.path.exists(args.config_path):
        config = loadModule(args.config_path)
    else:
        print('WARNING: Designated config file (%s) does not exist. Check the path. Exiting...'%args.config_path)
        sys.exit()

    # Generate run_description. If var argument is given, generate run_description out of it. If not, give it 'default'.
    orig = copyModule(config) # keep original information to track the changes to be made in the copied config.py file.
    if args.var:
        run_description = ''
        for var in args.var:
            dtype = type(getattr(orig, var[0]))
            if dtype.__name__ == 'tuple':
                # If tuple, assert the number of element is the same as in the original.
                assert(len(getattr(orig,var[0])) == len(var) - 1), "Expecting %s values but received %s."%(str(len(getattr(orig,var[0]))), str(len(var) - 1))
            if dtype.__name__ == 'tuple' or dtype.__name__ == 'list':
                # Get dtype of the first element of the variable in the original config file.
                element_dtype = type(getattr(orig, var[0])[0])
                # Create a list from the user-given values
                temp_list = [0,] * (len(var) - 1)
                for idx in range(len(var) - 1):
                    temp_list[idx] = element_dtype(var[idx+1])
                # Convert the list to a tuple and then assign to var[0] in configuration
                setattr(config, var[0], dtype(temp_list))
            else:
                assert(len(var) == 2), "Expecting a variable and value pair for variable %s."%var[0]
                setattr(config, var[0], dtype(var[1]))

            # Generate a run_description based on the variable value changes.
            for v in var:
                if len(run_description) > 0:
                    run_description = run_description + '_'
                run_description = '%s%s' % (run_description, str(v))
    else:
        run_description = 'default'

    if args.experiment_detail:
        run_description = args.experiment_detail            

    # Create a unique run_name out of run_description. It is run_description_[unique integer index for this configuration]
    try_cnt = 1
    run_name = run_description
    while os.path.exists(os.path.join(trainingOutputFolder(), 'results', args.experiment_name, run_name + '_code.zip')):
        run_name = run_description + '_' + str(try_cnt)
        try_cnt += 1

    # Generate paths for dumping result files and codes.
    dump_path = os.path.join(os.path.join(trainingOutputFolder(), 'results', args.experiment_name))
    code_dir_path = os.path.join(dump_path, run_name + '_code')

    # Identify changes made to config through the 'var' argument.
    config_diff = configDiff(orig, config)

    # Copy necessary files to results directory.
    copy_scripts(args.config_path, config_diff, code_dir_path)

    # Execute training depending on the mode.
    if args.mode == 'local':
        orig_working_dir = os.getcwd()
        os.chdir(os.path.join(trainingOutputFolder(),'results',args.experiment_name, run_name + '_code'))
        # provide trainPupil.py with appropriate dump path and run description so that it generates tensorboard log and torch weight files in the desired location.
        train_script_path = os.path.normpath(os.path.join(code_dir_path, 'trainPupil.py'))
        os.system('python %s -d %s -r %s'%(train_script_path, dump_path, run_name))
        # move back into the original working directory.
        os.chdir(orig_working_dir)
        # Remove the code directory.
        shutil.rmtree(code_dir_path)

    elif args.mode == 'remote':
        print('WARNING: remote training not supported')
