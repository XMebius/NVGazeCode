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

from tensorflow.python.summary.summary import FileWriter
from tensorflow.python.summary.summary import scalar, histogram, image
import tensorflow as tf
import numpy as np
import os, pdb, io, scipy

class SummaryWriter(object):
    def __init__(self, log_dir):
        
        self.file_writer = tf.summary.create_file_writer(log_dir, flush_millis=10000)

        self.steps = {}

    def step(self, type, name):
        k = type + "_" + name
        if k not in self.steps:
            self.steps[k] = 0
        self.steps[k] += 1

        return self.steps[k] - 1

    def add_scalar(self, name, scalar_value, global_step=None):
        if global_step is None:
            global_step = self.step("scalar", name)

        with self.file_writer.as_default(step=global_step):
            tf.summary.scalar(name, scalar_value, step=global_step)

    def add_histogram(self, name, values, global_step=None, bins = 100):
        if global_step is None:
            global_step = self.step("hist", name)

        cnts, bin_edges = np.histogram(values, bins = bins)
        # fill the fields of the hitogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        # dropping the start of the firsst bin
        bin_edges = bin_edges[1:]
        # add bin edges and counts
        for edge in bin_edges:
            hist.budket_limit.append(edge)
        for c in cnts:
            hist.bucket.append(c)

        with self.file_writer.as_default(step=global_step):
            tf.summary.histogram(name, hist)

    def add_images(self, tag, images, global_step=None):
        if global_step is None:
            global_step = self.step("image", tag)
        img_summaries = list()
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = io.StringIO()
            except:
                s = io.BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        self.file_writer.add_summary(tf.compat.v1.Summary(value=img_summaries), global_step)

    def add_image(self, tag, image, global_step=None):
        if global_step is None:
            global_step = self.step("image", tag)
        img_summaries = list()

        s = io.BytesIO()

        scipy.misc.toimage(image).save(s, format="png")
        # Create an Image object
        img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        # Create a Summary value
        img_summaries.append(tf.compat.v1.Summary.Value(tag='%s'%tag, image=img_sum))
        # Create and write Summary
        self.file_writer.add_summary(tf.compat.v1.Summary(value=img_summaries), global_step)

    def close(self):
        self.file_writer.flush()
        self.file_writer.close()

    def __del__(self):
        if self.file_writer is not None:
            self.file_writer.close()


class TensorBoardLogger(SummaryWriter):
    def __init__(self, active=True, experiment=None, base_log_dir="./logs"):

        if active:
            if experiment is None:
                experiment = os.path.basename(os.getcwd())

            i = 2
            temp_name = experiment
            while os.path.exists(os.path.join(base_log_dir, temp_name)):
                temp_name = experiment + "_" + str(i)
                i += 1

            print (temp_name)

            super(TensorBoardLogger, self).__init__(os.path.join(base_log_dir, temp_name))

        self.active = active

    def log_dict(self, dct):
        if self.active:
            for k in dct:
                self.add_scalar(k, dct[k])

    def log_image(self, dct):
        if self.active:
            for k in dct:
                self.add_image(k, dct[k])

    def log_dict_hist(self, dct):
        if self.active:
            for k in dct:
                self.add_histogram(k, dct[k])

    def add_scalar(self, name, scalar_value, global_step=None):
        if self.active:
            super(TensorBoardLogger, self).add_scalar(name, scalar_value, global_step)

    def add_histogram(self, name, values, global_step=None):
        if self.active:
            super(TensorBoardLogger, self).add_histogram(name, values, global_step)

    def add_images(self, name, images, global_step = None):
        if self.active:
            super(TensorBoardLogger, self).add_images(name, images, global_step)

    def add_image(self, name, image, global_step = None):
        if self.active:
            super(TensorBoardLogger, self).add_image(name, image, global_step)

    def __del__(self):
        if self.active:
            super(TensorBoardLogger, self).__del__()



