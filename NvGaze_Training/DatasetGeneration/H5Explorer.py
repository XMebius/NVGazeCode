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
Simple h5 file explorer
drag and drop h5 file into GUI
content of h5 file is then displayed and can be extractd into a defined folder
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import sys
import os
import platform
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pdb

import cv2

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
if op_sys == 'Darwin':
    from Foundation import NSURL

import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../.")
from utils import convertImageToGrayscale, convertGrayscaleToRGB, showImage


class MainWindowWidget(QWidget):
    """
    Subclass the widget and add a button to load an h5 File. 
    Alternatively allow drag'n'drop of h5 file onto the widget
    """

    def __init__(self):
        super(MainWindowWidget, self).__init__()

        # Button that allows loading of images
        self.load_button = QPushButton("Load h5 file")
        self.load_button.clicked.connect(self.load_h5_button)

        # viewing region
        self.lbl = QLabel(self)

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.load_button)
        layout_button.addStretch()

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        layout.addWidget(self.lbl)

        self.setLayout(layout)

        # Enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)

        self.show()
        
        self.h5file = None
        self.imagePtr = None
        self.imageIndex = 0

    def closeEvent(self, event):
        
        # terminate running threads and do other stuff

        if (self.h5file is not None):
            self.h5file.close()
            self.h5file = None
            self.imagePtr = None
            self.imageIndex = 0

    def load_h5_button(self):
        """
        Open a File dialog when the button is pressed
        :return:
        """
        
        #Get the file location
        self.h5filename, _ = QFileDialog.getOpenFileName(self, 'Open file', './', '*.h5')
        # Load the image from the location
        self.load_h5file()

    def showImageNoIndex(self):

            if (self.imagePtr is None):
                return

            print('image shape : ', self.imagePtr.shape)
            print('image dtype : ', self.imagePtr.dtype)

            image = np.asarray(self.imagePtr)

            if (len(self.imagePtr.shape) == 2):
                imageExp = np.expand_dims(image, axis=0)
                image = convertGrayscaleToRGB(np.asarray(imageExp))

            if (image.shape[0] == 1):
                image = convertGrayscaleToRGB(np.asarray(self.imagePtr));
            
            image = image.astype(np.uint8)
            image = np.transpose(image, [1,2,0])
            height = image.shape[0]
            width = image.shape[1]
            channels = image.shape[2]
            bytesPerLine = image.shape[2] * width
            qImg = QImage(bytes(image.data), width, height, bytesPerLine, QImage.Format_RGB888)
            
            pix = QPixmap(qImg)
            self.lbl.setPixmap(pix)

    def showImage(self, index):

        if (self.imagePtr is None):
            return

        ## get image by index
        # N E V C H W
        image = self.imagePtr[index,0,0] # type : numpy array (data in memory)
        
        print('sample index : ', index)
        print('image shape : ', image.shape)
        print('image dtype : ', image.dtype)

        if (image.shape[0] == 1):
            image = convertGrayscaleToRGB(image);
                       
        image = image.astype(np.uint8)
        image = np.transpose(image, [1,2,0])
        #print(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        bytesPerLine = image.shape[2] * width
        qImg = QImage(bytes(image.data), width, height, bytesPerLine, QImage.Format_RGB888)
        
        pix = QPixmap(qImg)
        self.lbl.setPixmap(pix)

    def showImageForCurrentIndex(self):

        # read images
        if (self.hasIndexedImages):
            if (self.imagePtr is not None):
                self.showImage(self.imageIndex)
            else:
                print('WARNING: imagePtr is none!')
        else:
            allKeys = list(self.h5f.keys())
            wrappedIndex = self.imageIndex % len(allKeys)
            print('showing image %d'%(wrappedIndex))
            key = allKeys[wrappedIndex]
            self.imagePtr = self.h5f[key]
            if (self.imagePtr is not None):
                self.showImageNoIndex()


    def dumpImage(self, index):

        if (self.imagePtr is None):
            return

        ## get image by index
        # N E V C H W
        image = self.imagePtr[index,0,0] # type : numpy array (data in memory)
        
        print('sample index : ', index)
        print('image shape : ', image.shape)
        print('image dtype : ', image.dtype)

        if (image.shape[0] == 1):
            image = convertGrayscaleToRGB(image);
                       
        image = image.astype(np.uint8)
        image = np.transpose(image, [1,2,0])
        
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        cv2.imwrite('imagedump.png', image)

    def load_h5file(self):
        """
        Set the image to the pixmap
        :return:
        """

        print(self.h5filename)
        binocular = False

        if (self.h5file is not None):
            self.h5file.close()
            self.h5file = None
            self.imagePtr = None
            self.imageIndex = 0

        # open h5 and show keys
        self.h5f = h5py.File(self.h5filename, 'r')
        
        print('keys')
        print(list(self.h5f.keys()))
        print('attrs.keys')
        print(list(self.h5f.attrs.keys()))

        for key in self.h5f.attrs.keys():
            print(key, self.h5f.attrs[key])
           
        if 'eyes' in self.h5f.attrs.keys():
            if (self.h5f.attrs['eyes'] == 'binocular'):
                binocular = True
           
        # read aperture masks, if any
        if 'aperture_masks' in self.h5f.keys():
            print('loading aperture masks')
            apertureMasks = self.h5f['aperture_masks'][:]
            print('aperture masks shape : ', apertureMasks.shape)
            print('aperture masks dtype : ', apertureMasks.dtype)
            singlemask = apertureMasks[0,0,0]
            showImage(singlemask)
        else:
                print('no aperture masks contained')

        # read region masks, if any
        if 'region_maps' in self.h5f.keys():
            print('loading region masks')
            regionmasks = self.h5f['region_maps'][:]
            print('region masks shape : ', regionmasks.shape)
            print('region masks dtype : ', regionmasks.dtype)
            singlemask = regionmasks[0,0,0,0]
        else:
            print('no aperture masks contained')

        # get image pointer for first image
        self.hasIndexedImages = False
        # check if we have a key containing all images
        for key in self.h5f.keys():
            if (key == 'images'):
                self.imagePtr = self.h5f[key]
                self.hasIndexedImages = True
        # otherwise we assume that all keys are pointing to images (with various sizes)

        if self.hasIndexedImages:
            print('dataset uses images key')
        else:
            print('dataset has no images key')

        self.showImageForCurrentIndex()


    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget
        File locations are stored in fname
        """
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
                if op_sys == 'Darwin':
                    fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
                else:
                    fname = str(url.toLocalFile())

            self.h5filename = fname
            self.load_h5file()
        else:
            e.ignore()

    def keyPressEvent(self, event):

        if (self.imagePtr is not None and self.imageIndex is not None):

            if event.key() == QtCore.Qt.Key_A:
                print ('previous image')
                self.imageIndex = self.imageIndex - 1
            if event.key() == QtCore.Qt.Key_D:
                print ('next image')
                self.imageIndex = self.imageIndex + 1
            if event.key() == QtCore.Qt.Key_W:
                self.imageIndex = self.imageIndex + 50
                print ('jump 50 forward')
            if event.key() == QtCore.Qt.Key_S:
                self.imageIndex = self.imageIndex - 50
                print ('jump 50 backward')
            if event.key() == QtCore.Qt.Key_Escape:
                self.deleteLater()
            if event.key() == QtCore.Qt.Key_Control:
                self.dumpImage(self.imageIndex) # write image

            event.accept()
            self.showImageForCurrentIndex()

# Run if called directly
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindowWidget()
    sys.exit(app.exec_())