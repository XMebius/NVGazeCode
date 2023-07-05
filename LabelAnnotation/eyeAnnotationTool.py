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

from __future__ import division, unicode_literals, print_function, absolute_import

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication, QCloseEvent, QColor
from PyQt5.QtCore import QPoint, QPointF, QRect
import sys
import os
import platform
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pdb
import cv2
import math

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
if op_sys == 'Darwin':
    from Foundation import NSURL

# import utils
import sys
import utils

class Ellipse():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.major = 0
        self.minor = 0
        self.angle = 0
        self.valid = False
    def ellipse(self, scalingFactor = 1.0):
        ellipse = ((self.x * scalingFactor, self.y * scalingFactor), (self.major * scalingFactor, self.minor * scalingFactor), self.angle)
        return ellipse

class SelectionPoint():
    def __init__(self, x = 0.0, y = 0.0, sel = False):
        self.x = x
        self.y = y
        self.selected = sel

    def select(self):
        self.selected = True

    def deselect(self):
        self.selected = False

    def distanceToPoint(self, px, py):
        return math.sqrt( (self.x - px)*(self.x - px) + (self.y - py)*(self.y - py) )
    
class ContourPoints():
    
    def __init__(self):
        self.points = []

    def clearPoints(self):
        self.points = []

    def addPoint(self,x,y,selected = True):
        self.points.append(SelectionPoint(x,y,selected))
    
    def removePointAtPosition(self,x,y,margin = 5.0):
        foundindices = []
        for idx in range(len(self.points)):
            if (self.points[idx].distanceToPoint(x,y) <= margin):
                foundindices.append(idx)
        # remove points, use reverse order so that indices stay valid
        for index in sorted(foundindices, reverse=True):
            del self.points[index]

    def selectPointsAtPosition(self,x,y,margin = 5.0):

        success = False
        for idx in range(len(self.points)):
            if (self.points[idx].distanceToPoint(x,y) <= margin):
                self.points[idx].select()
                success = True
            else:
                self.points[idx].deselect()
        return success

    def selectAll(self):
        for idx in range(len(self.points)):
            if (self.points[idx].selected == False):
                self.points[idx].select()


    def deselectPoints(self):
        for idx in range(len(self.points)):
            if (self.points[idx].selected):
                self.points[idx].deselect()

    def moveSelectedPoints(self,dx,dy):
        for idx in range(len(self.points)):
            if (self.points[idx].selected):
                self.points[idx].x = self.points[idx].x + float(dx)
                self.points[idx].y = self.points[idx].y + float(dy)

    def getPointList(self):
        pointlist = []
        for idx in range(len(self.points)):
            pointlist.append((self.points[idx].x, self.points[idx].y))
        return np.asarray(pointlist, dtype=np.float32)

    def drawContourPoints(self,img, color, scalingFactor):
        
        for p in self.points:
            center = (int(p.x * scalingFactor), int(p.y * scalingFactor))
            cv2.circle(img, center, 3, color, -1)

class ManualAnnotationTool():

    def __init__(self):
        self.pupilContourPoints = ContourPoints()
        self.irsContourPoints = ContourPoints()
        self.eyeballContourPoints = ContourPoints()

        self.method = 'inactive'
        self.active = False

        self.movingPoints  = False
        self.lastMousePosition = (0,0)

    def createPoint(self,x,y):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            self.pupilContourPoints.addPoint(x,y,False)

    def removePoint(self,x,y):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            self.pupilContourPoints.removePointAtPosition(x,y)

    def clearPoints(self):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            self.pupilContourPoints.clearPoints()
        if (self.method == 'iris'):
            self.irsContourPoints.clearPoints()
        if (self.method == 'eyeball'):
            self.eyeballContourPoints.clearPoints()
    
    def startMovingPoints(self,x,y,useAllPoints = False):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            startMoving = False
            if (useAllPoints):
                # select all points
                self.pupilContourPoints.selectAll()
                startMoving = True
            elif (self.pupilContourPoints.selectPointsAtPosition(x,y) == True):
                startMoving = True

            if (startMoving):
                self.movingPoints  = True
                self.lastMousePosition = (x,y)
                

    def movePoints(self,x,y):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            if (self.movingPoints):
                dx = x - self.lastMousePosition[0]
                dy = y - self.lastMousePosition[1]
                self.lastMousePosition = (x,y)
                self.pupilContourPoints.moveSelectedPoints(dx,dy)

    def stopMovingPoints(self):
        self.movingPoints  = False
        self.pupilContourPoints.deselectPoints()

    def toggle(self, method):
        if (method == 'pupil'):
            self.method = 'pupil'
            self.active = True
        elif (method == 'iris'):
            self.method = 'iris'
            self.active = True
        elif (method == 'eyeball'):
            self.method = 'eyeball'
            self.active = True
        else:
            self.method = 'inactive'
            self.active = False
        print('manual annotation tool mode = %s'%(self.method))

    def mouseClick(self, button, x, y):
        
        modifiers = QtGui.QGuiApplication.keyboardModifiers()

        if (button & QtCore.Qt.LeftButton):
            if modifiers == QtCore.Qt.ControlModifier:
                self.startMovingPoints(x,y,True)
            elif modifiers == QtCore.Qt.AltModifier:
                self.startMovingPoints(x,y)
            else:
                # create or select point
                if (self.pupilContourPoints.selectPointsAtPosition(x,y) == False):
                    self.createPoint(x,y)
                self.startMovingPoints(x,y)

        if (button & QtCore.Qt.RightButton):
            if modifiers == QtCore.Qt.ControlModifier:
                self.clearPoints()
            else:
                self.removePoint(x,y)

    def mouseRelease(self):
        self.stopMovingPoints()
        return

    def mouseMove(self, button, x,y):
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            if (self.movingPoints):
                self.movePoints(x,y)

    def draw(self, image, scalingFactor):
        
        if (self.active == False):
            return
        if (self.method == 'pupil'):
            self.pupilContourPoints.drawContourPoints(image, (255,0,0), scalingFactor)
            return
        elif (self.method == 'iris'):
            # draw contour points
            return
        elif (self.method == 'eyeball'):
            # draw contour points
            return

    def performFit(self, pupilLabels,irisLabels,eyeballLabels,imageIndex):

        ellipse = None

        success = False

        if (self.active == False):
            return False, ellipse

        if (self.method == 'pupil'):

            # get point list
            try:
                ellipse = cv2.fitEllipse(self.pupilContourPoints.getPointList())
                print(ellipse)
                success = True
            except:
                print('WARNING : ellipse fit failed')
                success = False

            if (success == True):
                # override old pupil ellipse
                pupilLabels[imageIndex].x = ellipse[0][0]
                pupilLabels[imageIndex].y = ellipse[0][1]
                pupilLabels[imageIndex].major = ellipse[1][0]
                pupilLabels[imageIndex].minor = ellipse[1][1]
                pupilLabels[imageIndex].angle = ellipse[2]
        #elif (self.method == 'iris'):
        #elif (self.method == 'eyeball'):

        return success, ellipse
    
class MainWindowWidget(QWidget):
    """
    Subclass the widget and add a button to load an h5 File. 
    Alternatively allow drag'n'drop of h5 file onto the widget
    """

    def __init__(self):
        super(MainWindowWidget, self).__init__()

        self.imageIndex = None
        self.imagePtr = None

        self.setWindowTitle('Eye Annotation tool (Copyright by NVIDIA Corp.)')

        # image viewing region
        self.lbl = QLabel(self)

        # Button that allows loading of images
        self.load_button = QPushButton("Load h5 file")
        self.load_button.clicked.connect(self.load_h5_button)

        # check box for histogram equalization [on,off]
        self.histeq_checkbox = QCheckBox("Hist Equalization")
        self.histeq_checkbox.stateChanged.connect(self.histeq_event)
        
        # slider for contrast [-1,1]
        self.contrast_slider = QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.valueChanged[int].connect(self.contrastslider_event)
        self.contrast_slider.setValue(50)

        self.histeq_checkbox.stateChanged.connect(self.histeq_event)

        # slider for brightness [-100,100]
        self.brightness_slider = QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.valueChanged[int].connect(self.brightnessslider_event)
        self.brightness_slider.setValue(50)

        # horizontal layout to include the button on the left
        layout_button = QVBoxLayout()
        layout_button.addWidget(QLabel("Brightness"))
        layout_button.addWidget(self.brightness_slider)
        layout_button.addWidget(QLabel("Contrast"))
        layout_button.addWidget(self.contrast_slider)
        layout_button.addWidget(self.histeq_checkbox)
        layout_button.addStretch()

        layout_image = QVBoxLayout()
        layout_image.addWidget(self.lbl)

        # A Vertical layout to include the button layout and then the image
        layout = QHBoxLayout()
        layout.addLayout(layout_image)
        layout.addLayout(layout_button)
        
        layout.setAlignment (QtCore.Qt.AlignTop )
        layout_image.setAlignment (QtCore.Qt.AlignTop )
        layout_button.setAlignment (QtCore.Qt.AlignTop )

        # set layout
        self.setLayout(layout)

        # Enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)

        self.show()
        
        self.performHistEq      = self.histeq_checkbox.isChecked()
        self.brightnessOffset   = 0
        self.contrast           = 1.0

        self.scalingFactor      = 2.0

        self.h5file = None

        self.labelsDirty = None
        self.uncheckedSamples = None

    def saveDatabase(self):

        success = False

        performSave = False

        if (self.h5file is None):
            return

        if (self.labelsDirty is not None):
            if (self.labelsDirty == False):
                buttonReply = QMessageBox.question(self, 'PyQt5 message', "No labels have been changed. Do you still want to save ?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if buttonReply == QMessageBox.Yes:
                    performSave = True
                if buttonReply == QMessageBox.No:
                    performSave = False
                    success = True
                if buttonReply == QMessageBox.Cancel:
                    performSave = False
                    success = False
            else:
                if (len(self.uncheckedSamples) > 0):
                    buttonReply = QMessageBox.question(self, 'PyQt5 message', "Not all labels have been checked. Do you still want to save ?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                    if buttonReply == QMessageBox.Yes:
                        performSave = True
                    if buttonReply == QMessageBox.No:
                        performSave = False
                        success = True
                    if buttonReply == QMessageBox.Cancel:
                        success = False
                        performSave = False

        if (performSave == True and self.h5file is not None):
            print('saving')
            success = True

            # load all images
            allImages = self.h5file['images'][:]

            self.h5file.close()
            self.h5file = None
            self.imagePtr = None
            self.imageIndex = 0
            
            with h5py.File(self.h5filename, 'r+') as h5f:

                if self.titlePupilX not in h5f.keys():
                    h5f.create_dataset(self.titlePupilX, [len(self.pupilLabels)], dtype=np.float32)
                if self.titlePupilY not in h5f.keys():
                    h5f.create_dataset(self.titlePupilY, [len(self.pupilLabels)], dtype=np.float32)
                if self.titlePupilMajorRadius not in h5f.keys():
                    h5f.create_dataset(self.titlePupilMajorRadius, [len(self.pupilLabels)], dtype=np.float32)
                if self.titlePupilMinorRadius not in h5f.keys():
                    h5f.create_dataset(self.titlePupilMinorRadius, [len(self.pupilLabels)], dtype=np.float32)
                if self.titlePupilAngle not in h5f.keys():
                    h5f.create_dataset(self.titlePupilAngle, [len(self.pupilLabels)], dtype=np.float32)

                # overwrite only data that has been accepted
                for idx in self.acceptedSamples:
                    h5f[self.titlePupilX][idx]            = self.pupilLabels[idx].x
                    h5f[self.titlePupilY][idx]            = self.pupilLabels[idx].y
                    h5f[self.titlePupilMajorRadius][idx]  = self.pupilLabels[idx].major
                    h5f[self.titlePupilMinorRadius][idx]  = self.pupilLabels[idx].minor
                    h5f[self.titlePupilAngle][idx]        = self.pupilLabels[idx].angle

                if 'validSamples' in h5f.keys():
                    del h5f['validSamples']
                dataset = h5f.create_dataset("validSamples", [len(self.acceptedSamples)], dtype=np.int32)
                dataset[:] = np.asarray(self.acceptedSamples, dtype=np.int32)

                if 'rejectedSamples' in h5f.keys():
                    del h5f['rejectedSamples']
                dataset = h5f.create_dataset("rejectedSamples", [len(self.rejectedSamples)], dtype=np.int32)
                dataset[:] = np.asarray(self.rejectedSamples, dtype=np.int32)

                print('new h5 file written')
                success = True

        return success

    def closeEvent(self, event):
        
        # terminate your threads and do other stuff
        performQuit = True

        if (self.labelsDirty is not None and self.uncheckedSamples is not None):
            if (self.labelsDirty == True or len(self.uncheckedSamples) > 0):
                performQuit = self.saveDatabase()
            
        if (performQuit):
            print('Shutting down application')            
            if (self.h5file is not None):
                self.h5file.close()
                self.h5file = None
                self.imagePtr = None
                self.imageIndex = 0

            event.accept()
        else:
            event.ignore()

    def load_h5_button(self):
        """
        Open a File dialog when the button is pressed
        :return:
        """
        
        #Get the file location
        self.h5filename, _ = QFileDialog.getOpenFileName(self, 'Open file', './', '*.h5')
        # Load the image from the location
        self.load_h5file()

    def updateWindow(self):
        if (self.imagePtr is not None):
            self.showImage(self.imageIndex)

    def contrastslider_event(self, value):
        self.contrast = (float(value) / 100.0) * 2.0
        self.updateWindow()

    def brightnessslider_event(self, value):
        self.brightnessOffset = (float(value) / 100.0 - 0.5) * 200.0
        self.updateWindow()
        
    def histeq_event(self, state):
        if state == 2:
            self.performHistEq = True
        else:
            self.performHistEq = False
        self.updateWindow()

    def showImage(self, index):

        if (self.imagePtr is None):
            return

        col = QColor(0, 0, 255)
        if (index in self.uncheckedSamples):
            col = QColor(100, 100, 0)
        elif (index in self.acceptedSamples):
             col = QColor(0, 100, 0) # accepted
        elif (index in self.rejectedSamples):
             col = QColor(100, 0, 0) # accepted

        p = self.palette()
        p.setColor(self.backgroundRole(), col) # reject
        self.setPalette(p)

        ## get image by index
        # N E V C H W
        image = self.imagePtr[index,0,0] # type : numpy array (data in memory)
        image = np.squeeze(image,0)
        
        # apply brightness offset
        image = np.asarray(np.clip(np.asarray(image, dtype=np.int32) + self.brightnessOffset,0,255), dtype=np.uint8)
        
        # contrast scaling
        cv2.convertScaleAbs(image, image, self.contrast, 0 );

        if self.performHistEq:
            image = cv2.equalizeHist(image)

        if (len(image.shape) == 2):
            image = np.expand_dims(image,0)

        if (image.shape[0] == 1):
            image = utils.convertGrayscaleToRGB(image);
  
        image = np.transpose(image, [1,2,0])
        image = image.astype(np.uint8)
        image = image.copy() # important !!!!

        image = cv2.resize(image,dsize=None,fx=self.scalingFactor,fy=self.scalingFactor)
       
        # draw pupil ellipse
        if (self.pupilLabels is not None):
            ellipse = self.pupilLabels[self.imageIndex]
            #if (ellipse.valid):
            e = self.pupilLabels[self.imageIndex].ellipse(self.scalingFactor)
            cv2.ellipse(image,e,(0,255,0),1)
            cv2.circle(image,(int(e[0][0]),int(e[0][1])),1,(0,255,0),2)

        ## iris ellipse
        #if (self.irisLabels is not None):
        #    e = self.irisLabels[self.imageIndex].ellipse()
        #    cv2.ellipse(image,e,(0,255,255),1)
        #    cv2.circle(image,(int(e[0][0]),int(e[0][1])),1,(0,255,255),2)

        ## eyeball ellipse
        #if (self.eyeballLabels is not None):
        #    e = self.eyeballLabels[self.imageIndex].ellipse()
        #    cv2.ellipse(image,e,(255,0,255),1)
        #    cv2.circle(image,(int(e[0][0]),int(e[0][1])),1,(0,255,255),2)
        
        # draw annotation tool
        if (self.manualAnnotationTool.active):
            self.manualAnnotationTool.draw(image, self.scalingFactor)

        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        bytesPerLine = image.shape[2] * width
        qImg = QImage(bytes(image.data), width, height, bytesPerLine, QImage.Format_RGB888)
        
        pix = QPixmap(qImg)
        self.lbl.setPixmap(pix)

    def dumpImage(self, index):

        if (self.imagePtr is None):
            return

        ## get image by index
        # N E V C H W
        image = self.imagePtr[index,0,0] # type : numpy array (data in memory)
        
        print('sample index : ', index)

        if (image.shape[0] == 1):
            image = utils.convertGrayscaleToRGB(image);
                       
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

        self.imageIndex = 0

        def initializePupilLabels(numSamples):
            # get number of samples
            self.pupilLabels = []
            for i in range(numSamples):
                ellipse = Ellipse()
                self.pupilLabels.append(ellipse)
            print('initialized %d pupil labels'%(numSamples))

        def initializeIrisLabels(numSamples):
            # get number of samples
            self.irisLabels = []
            for i in range(numSamples):
                ellipse = Ellipse()
                self.irisLabels.append(ellipse)
            print('initialized %d iris labels'%(numSamples))        

        def initializeEyeballLabels(numSamples):
            # get number of samples
            self.eyeballLabels = []
            for i in range(numSamples):
                ellipse = Ellipse()
                self.eyeballLabels.append(ellipse)
            print('initialized %d iris labels'%(numSamples))        

        print(self.h5filename)
        binocular = False

        if (self.h5file is not None):
            self.h5file.close()
            self.h5file = None
            self.imagePtr = None
            self.imageIndex = 0

        # open h5 and show keys
        self.h5file = h5py.File(self.h5filename, 'r')
        
        print(list(self.h5file.keys()))
        print(list(self.h5file.attrs.keys()))

        self.pupilLabels = None
        self.irisLabels = None
        self.eyeballLabels = None

        self.validSamples = None

        self.titlePupilX                    = 'float_pupil_x'
        self.titlePupilY                    = 'float_pupil_y'
        self.titlePupilMajorRadius          = 'float_pupilellipse_major'
        self.titlePupilMinorRadius          = 'float_pupilellipse_minor'
        self.titlePupilAngle                = 'float_pupilellipse_angle'

        self.titleIrisX                    = 'float_iris_x'
        self.titleIrisY                    = 'float_iris_y'
        self.titleIrisMajorRadius          = 'float_irisllipse_major'
        self.titleIrisMinorRadius          = 'float_irisellipse_minor'
        self.titleIrisAngle                = 'float_irisellipse_angle'

        self.titleEyeballX                  = 'float_eyeball_x'
        self.titleEyeballY                  = 'float_eyeball_y'
        self.titleEyeballDiameter           = 'float_eyeball_diameter'

        self.numSamples = None

        self.labelsDirty = False

        for key in self.h5file.keys():
            
            print(key)

            # read images
            if (key == 'images'):
                self.imagePtr = self.h5file[key] # type : h5 dataset (data on disc)
                self.numSamples = self.imagePtr.shape[0]
                print('num samples', self.numSamples)
                

        if (self.numSamples is None):
            print('ERROR: no images contained in h5 files')
            return

        for key in self.h5file.keys():

            # import pupil ellipse
            if (key == self.titlePupilX):
                if (self.pupilLabels is None):
                    initializePupilLabels(self.numSamples)
                for idx, p in enumerate(self.pupilLabels):
                    p.x = float(self.h5file[key][idx])
                    p.valid = True

            if (key == self.titlePupilY):
                if (self.pupilLabels is None):
                    initializePupilLabels(self.numSamples)
                for idx, p in enumerate(self.pupilLabels):
                    p.y = float(self.h5file[key][idx])
                    p.valid = True

            if (key == self.titlePupilMajorRadius):
                if (self.pupilLabels is None):
                    initializePupilLabels(self.numSamples)
                for idx, p in enumerate(self.pupilLabels):
                    p.major = float(self.h5file[key][idx])
                    p.valid = True

            if (key == self.titlePupilMinorRadius):
                if (self.pupilLabels is None):
                    initializePupilLabels(self.numSamples)
                for idx, p in enumerate(self.pupilLabels):
                    p.minor = float(self.h5file[key][idx])
                    p.valid = True

            if (key == self.titlePupilAngle):
                if (self.pupilLabels is None):
                    initializePupilLabels(self.numSamples)
                for idx, p in enumerate(self.pupilLabels):
                    p.angle = float(self.h5file[key][idx])
                    p.valid = True

        if (self.pupilLabels is None):
            print('no pupil labels contained in h5. creating empty labels')
            initializePupilLabels(self.numSamples)
        if (self.irisLabels is None):
            print('no iris labels contained in h5. creating empty labels')
            initializeIrisLabels(self.numSamples)
        if (self.eyeballLabels is None):
            print('no eyeball labels contained in h5. creating empty labels')
            initializeEyeballLabels(self.numSamples)

        # set all samples to valid / to be exported
        self.uncheckedSamples = list(range(self.numSamples))
        self.acceptedSamples = []
        self.rejectedSamples = []
        
        if 'validSamples' in self.h5file.keys():
            valid = self.h5file['validSamples'][:]

            for idx in valid:
                self.uncheckedSamples.remove(idx)
                self.acceptedSamples.append(idx)
            print('valid samples')
            print(self.acceptedSamples)

        if 'rejectedSamples' in self.h5file.keys():
            rejected = self.h5file['rejectedSamples'][:]
            for idx in rejected:
                if (idx in self.uncheckedSamples):
                    self.uncheckedSamples.remove(idx)
                    self.rejectedSamples.append(idx)
            print('rejected samples')
            print(self.rejectedSamples)

        if 'eyes' in self.h5file.attrs.keys():
            if (self.h5file.attrs['eyes'] == 'binocular'):
                binocular = True
           
        # read aperture masks, if any
        if 'aperture_masks' in self.h5file.keys():
            print('loading aperture masks')
            apertureMasks = self.h5file['aperture_masks'][:]
            print('aperture masks shape : ', apertureMasks.shape)
            print('aperture masks dtype : ', apertureMasks.dtype)
            singlemask = apertureMasks[0,0,0]
            showImage(singlemask)
        else:
                print('no aperture masks contained')

        # manual selection tool
        self.manualAnnotationTool = ManualAnnotationTool()
        
        # show first image
        if (self.imagePtr is not None):
            self.showImage(self.imageIndex)

    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    # drag move
    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
    
    # drop event
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

    def mousePressEvent(self, event):

        if (self.manualAnnotationTool.active):
            
            # get position in image space
            px = event.pos().x() - self.lbl.geometry().x()
            py = event.pos().y()- self.lbl.geometry().y()

            # check if position is inside geometry
            if (px < 0 or py < 0 or px > self.lbl.geometry().width() or py > self.lbl.geometry().height()):
                return
            px = float(px) / self.scalingFactor
            py = float(py) / self.scalingFactor

            if event.buttons() & QtCore.Qt.LeftButton:
                    self.manualAnnotationTool.mouseClick(QtCore.Qt.LeftButton, px, py )
            if event.buttons() & QtCore.Qt.RightButton:
                    self.manualAnnotationTool.mouseClick(QtCore.Qt.RightButton, px, py )

        self.updateWindow()

    def mouseReleaseEvent(self, event):
        if (self.manualAnnotationTool.active):
            self.manualAnnotationTool.mouseRelease()
        self.updateWindow()

    def mouseMoveEvent(self, event):
        # get position in image space
        px = event.pos().x() - self.lbl.geometry().x()
        py = event.pos().y()- self.lbl.geometry().y()
        px = float(px) / self.scalingFactor
        py = float(py) / self.scalingFactor
        if (self.manualAnnotationTool.active):
            self.manualAnnotationTool.mouseMove(event.buttons(),px,py)
        self.updateWindow()

    def keyPressEvent(self, event):

        if (self.imagePtr is not None and self.imageIndex is not None):

            if event.key() == QtCore.Qt.Key_1:
                self.manualAnnotationTool.toggle('pupil')
            if event.key() == QtCore.Qt.Key_2:
                self.manualAnnotationTool.toggle('iris')
            if event.key() == QtCore.Qt.Key_3:
                self.manualAnnotationTool.toggle('eyeball')
            if event.key() == QtCore.Qt.Key_0:
                self.manualAnnotationTool.toggle('inactive')

            ## function key
            if event.key() == QtCore.Qt.Key_Shift:
                if (self.manualAnnotationTool.active):
                    success = self.manualAnnotationTool.performFit(self.pupilLabels,self.irisLabels,self.eyeballLabels,self.imageIndex)
                    if (success):
                        self.labelsDirty = True
                        if (self.imageIndex in self.rejectedSamples):
                            self.rejectedSamples.remove(self.imageIndex)
                            self.acceptedSamples.append(self.imageIndex)                        
                        if (self.imageIndex in self.uncheckedSamples):
                            self.uncheckedSamples.remove(self.imageIndex)
                            self.acceptedSamples.append(self.imageIndex)                        


            # reject sample
            if event.key() == QtCore.Qt.Key_R:
                if (self.imageIndex in self.acceptedSamples):
                    self.acceptedSamples.remove(self.imageIndex)
                    self.rejectedSamples.append(self.imageIndex)                        
                if (self.imageIndex in self.uncheckedSamples):
                    self.uncheckedSamples.remove(self.imageIndex)
                    self.rejectedSamples.append(self.imageIndex)           

            if event.key() == QtCore.Qt.Key_T:
                if (self.imageIndex in self.rejectedSamples):
                    self.rejectedSamples.remove(self.imageIndex)
                    self.acceptedSamples.append(self.imageIndex)                        
                if (self.imageIndex in self.uncheckedSamples):
                    self.uncheckedSamples.remove(self.imageIndex)
                    self.acceptedSamples.append(self.imageIndex)                       


            if event.key() == QtCore.Qt.Key_A:
                self.imageIndex = max(0, self.imageIndex - 1)
                print ("sample %d"%(self.imageIndex))
            if event.key() == QtCore.Qt.Key_D:
                self.imageIndex = min(self.imageIndex + 1, self.numSamples-1)
                print ("sample %d"%(self.imageIndex))
            if event.key() == QtCore.Qt.Key_W:
                self.imageIndex = min(self.imageIndex + 50, self.numSamples-1)
                print ("sample %d"%(self.imageIndex))
            if event.key() == QtCore.Qt.Key_S:
                self.imageIndex = max(0, self.imageIndex - 50)
                print ("sample %d"%(self.imageIndex))
            if event.key() == QtCore.Qt.Key_Escape:
                self.close()
            if event.key() == QtCore.Qt.Key_P:
                modifiers = QtGui.QGuiApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ControlModifier:
                    self.dumpImage(self.imageIndex) # write image

            event.accept()
            self.updateWindow()

# Run if called directly
if __name__ == '__main__':
    app = QApplication(sys.argv) # Initialize the application
    ex = MainWindowWidget() # Call the widget
    sys.exit(app.exec_())