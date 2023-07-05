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

import numpy as np
import os
import pdb
import h5py
import cv2
import math
import zipfile
import utils

def extractPupilImage(img, pupilPosition, pupilRegionSize):

    pupilRegionSizeHalf = [math.floor(pupilRegionSize[0] / 2.0), math.floor(pupilRegionSize[1] / 2.0)]

    # get pupil region of interest
    x = math.floor(pupilPosition[0])
    y = math.floor(pupilPosition[1])

    imgPupilROI = [x - pupilRegionSizeHalf[0],
                    x - pupilRegionSizeHalf[0] + pupilRegionSize[0],
                    y - pupilRegionSizeHalf[1],
                    y - pupilRegionSizeHalf[1] +  pupilRegionSize[1]] # RECT

    success = True

    if (imgPupilROI[0] < 0 or imgPupilROI[2] < 0 or imgPupilROI[3] >= resY or imgPupilROI[1] >= resX):
        print(imgPupilROI)
        print("WARNING: skipping sample %d due to ill-posed border condition"%(sampleIdx))

        success = False

        if (False):

            visImage = np.reshape(img,[288,384, 1])
            visImage = np.repeat(visImage, 3, axis=2)
                
            #visImagePupil = np.reshape(pupilImg,[imgPupilROI[3], imgPupilROI[2], 1])
            #visImagePupil = np.repeat(visImagePupil, 3, axis=2)
        
            # draw pupil location
            cv2.circle(visImage,(int(pupilXConverted[sampleIdx]), int(pupilYConverted[sampleIdx])), 2, (0,255,0), -1)
            #cv2.circle(visImagePupil,(int(pupilRegionSizeHalf[0]), int(pupilRegionSizeHalf[1])), 2, (0,255,0), -1)

            # show image
            cv2.imshow('img',visImage)
            #cv2.imshow('pupil',visImagePupil)
            k = cv2.waitKey(0)
            #print (k)
            
    pupilImg = None
    if success:
        # get image of centered pupil location
        # cut out pupil region with given size
        pupilImg = img[0,0,0, imgPupilROI[2]:imgPupilROI[3], imgPupilROI[0]:imgPupilROI[1]]

    return pupilImg, success

def getMaskInRange(map, color, margin):

    # split map 
    b,g,r = cv2.split(np.asarray(map, dtype=np.float32))
    
    bInv = np.full(np.shape(b), 255, dtype=np.float32) - b
    gInv = np.full(np.shape(g), 255, dtype=np.float32) - g
    rInv = np.full(np.shape(r), 255, dtype=np.float32) - r

    # threshold pupil area
    mask = np.zeros(np.shape(regionmap), dtype=np.uint8)
    
    bMin = (color[0] - margin)
    bMax = (255 - (color[0] + margin))

    gMin = (color[1] - margin)
    gMax = (255 - (color[1] + margin))

    rMin = (color[2] - margin)
    rMax = (255 - (color[2] + margin))

    # compute geometric mean of pupil area
    ret,maskGreater0 = cv2.threshold(b, bMin, 255, cv2.THRESH_BINARY)
    ret,maskSmaller0 = cv2.threshold(bInv, bMax, 255, cv2.THRESH_BINARY)
    
    ret,maskGreater1 = cv2.threshold(g, gMin, 255, cv2.THRESH_BINARY)
    ret,maskSmaller1 = cv2.threshold(gInv, gMax, 255, cv2.THRESH_BINARY)

    ret,maskGreater2 = cv2.threshold(r, rMin, 255, cv2.THRESH_BINARY)
    ret,maskSmaller2 = cv2.threshold(rInv, rMax, 255, cv2.THRESH_BINARY)

    # compute combined mask
    mask = maskGreater0
    #pdb.set_trace()
    mask = cv2.bitwise_and(mask,maskSmaller0)
    mask = cv2.bitwise_and(mask,maskGreater1)
    mask = cv2.bitwise_and(mask,maskSmaller1)
    mask = cv2.bitwise_and(mask,maskGreater2)
    mask = cv2.bitwise_and(mask,maskSmaller2)

    return mask

def estimatePupilEllipse(regionmap):

    success = False
    pupilCenter = None
    pupilEllipse = None

    color = [0,255,0]
    margin = 10

    # derive mask
    mask = np.asarray(getMaskInRange(regionmap, color, margin), dtype=np.uint8)

    # perform ellipse fitting for mask
    contours = cv2.findContours(mask, 1, 2)
    ellipse = None

    #print(contours)
    #cv2.imshow('pupilMask', mask)
    #cv2.waitKey(0)

    for c in contours:
        
        try:
            for thisC in c:
                #print('length ',len(thisC))
                #print('sum ', np.sum(thisC))
                if (np.sum(thisC) <= 0 or len(thisC) < 5):
                    continue

                # debug
                #cv2.drawContours(mask, thisC, -1, (255,0,0), 3)

                try:
                    ellipse = cv2.fitEllipse(thisC)
                except:
                    #print('error on ellipse computation\n')
                    continue

        except:
            #print('error on drawing contour\n')
            continue
    
    #cv2.ellipse(mask,ellipse,(0,255,0),2)

    if (ellipse is not None):
        success = True

        pupilEllipse = ellipse
        pupilCenter = (ellipse[0][0], ellipse[0][1])

        #pdb.set_trace()

    #print(success)

    # cv2.imshow("pupilellipse", regionmap)
    # cv2.imshow("mask", mask)
    # cv2.ellipse(img,ellipse,(255,255,255),1)
    # cv2.circle(img,(int(ellipse[0][0]),int(ellipse[0][1])),1,(0,0,255),2)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    return success, pupilCenter, pupilEllipse

def estimateIrisEllipse(regionmap):

    color = [0,255,255] # iris
    margin = 10

    success = False
    irisCenter = None
    irisEllipse = None

    # derive mask
    mask = np.asarray(getMaskInRange(regionmap, color, margin), dtype=np.uint8)

    # perform ellipse fitting for mask
    contours = cv2.findContours(mask, 1, 2)
    ellipse = None

    #print(contours)
    #cv2.imshow('irisMask', mask)
    #cv2.waitKey(0)

    #print(contours)

    for c in contours:

        try:
            for thisC in c:
                if (np.sum(thisC) <= 0 or len(thisC) < 5):
                    continue
                #debug
                #cv2.drawContours(mask, thisC, -1, (255,0,0), 3)
                try:
                    ellipse = cv2.fitEllipse(thisC)
                except:
                    continue

        except:
            continue

    if (ellipse is not None):
        success = True

        irisEllipse = ellipse
        irisCenter = (ellipse[0][0], ellipse[0][1])

    # cv2.imshow("regionmap", regionmap)
    # cv2.imshow("mask", mask)
    # cv2.ellipse(img,irisEllipse,(0,255,255),1)
    # cv2.circle(img,(int(irisCenter[0]),int(irisCenter[1])),1,(0,255,255),2)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    return success, irisCenter, irisEllipse

def estimateEyeBall(img,regionmap):

    #color = [228,127,238]
    color = [0,217,255] # eyeball color
    margin = 100

    success = False
    eyeballCenter = None
    eyeballEllipse = None

    # derive mask
    mask = np.asarray(getMaskInRange(regionmap, color, margin), dtype=np.uint8)

    # perform ellipse fitting for mask
    contours = cv2.findContours(mask, 1, 2)
    ellipse = None

    # iterate over extracted contours
    for c in contours:
        try:
            for thisC in c:
                if (np.sum(thisC) <= 0 or len(thisC) < 5):
                    continue
                # debug
                #cv2.drawContours(mask, thisC, -1, (255,0,0), 3)
                try:
                    ellipse = cv2.fitEllipse(thisC)
                except:
                    continue
        except:
            continue

    if (ellipse is not None):
        success = True

        eyeballEllipse = ellipse
        eyeballCenter = (ellipse[0][0], ellipse[0][1])

        #print(eyeballEllipse)

    # cv2.imshow("regionmap", regionmap)
    # cv2.imshow("mask", mask)
    # cv2.ellipse(img, eyeballEllipse,(0,255,255),1)
    # cv2.circle(img,(int(eyeballCenter[0]),int(eyeballCenter[1])),1,(0,255,255),2)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    return success, eyeballCenter, eyeballEllipse

def scaleEllipse(ellipse, horScale, vertScale):
    
    majorRadius = ellipse[1][0]
    minorRadius = ellipse[1][1]
    angleRad = math.radians(ellipse[2])

    # rotate by angle
    majorx = majorRadius * math.cos(angleRad) - 0.0 * math.sin(angleRad)
    majory = majorRadius * math.sin(angleRad) + 0.0 * math.cos(angleRad)

    minorx = 0.0 * math.cos(angleRad) - minorRadius * math.sin(angleRad)
    minory = 0.0 * math.sin(angleRad) + minorRadius * math.cos(angleRad)

    # scale
    majorx = majorx * horScale
    majory = majory * vertScale

    minorx = minorx * horScale
    minory = minory * vertScale
                
    # rotate back
    majorRadius = majorx * math.cos(-angleRad) - majory * math.sin(-angleRad)
    minorRadius = minorx * math.sin(-angleRad) + minory * math.cos(-angleRad)

    ellipse = (ellipse[0], (majorRadius,minorRadius), ellipse[2])

    return ellipse

def scaleEllipseNoAngle(ellipse, horScale, vertScale):
    
    majorRadius = ellipse[1][0]
    minorRadius = ellipse[1][1]

    # scale
    majorRadius = majorRadius * horScale
    minorRadius = minorRadius * vertScale

    ellipse = (ellipse[0], (majorRadius,minorRadius), ellipse[2])

    return ellipse

if __name__ == '__main__':


    # configuration

    
    pupilRegionSize                 = [127, 127]
    rejectIncompletePupilsAtBorder  = True
    interactive                     = False

    rawDataFolderList = [
        'D:/NvGaze/Datasets/raw/nxp_male_01_in'
        ]

    outputFootageFolderList = [
        'D:/NvGaze/Datasets/raw/nxp_male_01_out',
    ]

    # specify output aspect ratio
    
    #outputResX = 293.0
    #outputResY = 293.0

    outputResX = 320.0
    outputResY = 240.0
    
    #outputResX = 640.0
    #outputResY = 480.0


    aspect = outputResY/outputResX
    resX = 320.0
    resY = resX * aspect

    # for each combination of data folder and output footage folder
    for datafolder, footagefolder in zip(rawDataFolderList,outputFootageFolderList):
        
        print ('%s, %s\n'%(datafolder, footagefolder))

        # create output folder if not exisiting
        if (os.path.exists(footagefolder) == False):
            print('creating footage folder %s'%(footagefolder))
            os.mkdir(footagefolder)

        # read CSV file
        csvfile = os.path.join(datafolder, 'footage_description.csv')
        (csv_header, numSamples) = utils.importCSVHeader(csvfile)
        (labels, labellist, numSamples, validLabels) = utils.parseCSV(csvfile)
       
        #print(labels['image_L_0'])
        imageLabels             = labels['image_L_0']
        regionmaskLabels        = labels['regionmask_withoutskin_L_0']
        regionmaskSkinLabels    = labels['regionmask_withskin_L_0']

        shape = np.shape(labels['float_gaze_x_degree'])
        labels['float_pupil_x'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_pupil_x')

        labels['float_pupil_y'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_pupil_y')

        labels['float_pupilellipse_major'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_pupilellipse_major')

        labels['float_pupilellipse_minor'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_pupilellipse_minor')

        labels['float_pupilellipse_angle'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_pupilellipse_angle')
        
        labels['float_iris_x'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_iris_x')
        
        labels['float_iris_y'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_iris_y')

        labels['float_irisellipse_major'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_irisellipse_major')

        labels['float_irisellipse_minor'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_irisellipse_minor')

        labels['float_irisellipse_angle'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_irisellipse_angle')

        labels['float_eyeball_x'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_eyeball_x')

        labels['float_eyeball_y'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_eyeball_y')

        labels['float_eyeball_diameter'] = np.zeros(shape, dtype=np.float32)
        labellist.append('float_eyeball_diameter')
        

        # output zip
        outzip = os.path.join(footagefolder, 'footage_image_data.zip')

        validSamples = []

        with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED) as zw:
        
            # for each image and region map pair
            print('processing archive %s'%(zw))
            
            for idx in range(numSamples):

                print('processing frame %d'%(idx))
        
                # check if image and region map exist
                imgFile = os.path.join(datafolder, str(imageLabels[idx]))
                regionmaskFile = os.path.join(datafolder, str(regionmaskLabels[idx]))
                #regionmaskSkinFile = os.path.join(datafolder, str(regionmaskSkinLabels[idx]))

                if (os.path.exists(imgFile) == False or os.path.exists(regionmaskFile) == False):# or os.path.exists(regionmaskSkinFile) == False):
                    print('\nERROR: image or region map files not existing for sample %d. stopping conversion for footage folder\n'%(idx))
                    print(imgFile)
                    print(regionmaskFile)
                    #print(regionmaskSkinFile)
                    break

                # load image
                img = cv2.imread(imgFile)
                img = cv2.resize(img, (int(resX),int(resY)))

                # load region map skin
                #regionmapskin = cv2.imread(regionmaskSkinFile)
                #regionmapskin = cv2.resize(regionmapskin,  (int(resX),int(resY)))
            
                # load region map
                regionmap = cv2.imread(regionmaskFile)
                regionmapUnscaled = regionmap.copy() # save unscaled regionmap for eye ball circle
                regionmap = cv2.resize(regionmap, (int(resX),int(resY)))

                # estimate output information : pupil, iris, eyeball
                (pupilsuccess, pupilCenter, pupilEllipse) = estimatePupilEllipse(regionmap)
                if (pupilsuccess == False):
                    print('pupil estimation failed')
                    continue

                (irissuccess, irisCenter, irisEllipse) = estimateIrisEllipse(regionmap)
                if (irissuccess == False):
                    print('iris estimation failed')
                    continue

                (eyeballsuccess, eyeballcenter, eyeballEllipse) = estimateEyeBall(img,regionmap)
                if (eyeballsuccess == False):
                    print('eyeball estimation failed')
                    continue                        

                # scale eyeball
                eyeballcenter = (eyeballcenter[0] * resX / float(np.shape(regionmapUnscaled)[1]), eyeballcenter[1] * resY / float(np.shape(regionmapUnscaled)[0]))
                eyeballEllipse = (eyeballcenter, (eyeballEllipse[1][0] * resX / float(np.shape(regionmapUnscaled)[1]), eyeballEllipse[1][1] * resY / float(np.shape(regionmapUnscaled)[0])), eyeballEllipse[2])


                # optional : perform image + label crop
                # copy image to output footage directory

                fraction = 51.0 / 60.0
                croppedWidth = int(fraction * resX)
                croppedHeight = int(fraction * resY)
                offset = (int((1.0 - fraction) / 2.0 * resX), int((1.0 - fraction) / 2.0 * resY))

                # crop image
                croppedImage = img[offset[1]:offset[1]+croppedHeight,offset[0]:offset[0]+croppedWidth,...].copy()
                # resize to output resolution
                resizedImage = cv2.resize(croppedImage, (int(outputResX),int(outputResY)))
                

                newImgFilePath = './resizedImage.png'
                cv2.imwrite(newImgFilePath, resizedImage)

                # crop region map without skin
                croppedRegionMap = regionmap[offset[1]:offset[1]+croppedHeight,offset[0]:offset[0]+croppedWidth,...].copy()
                # resize to output resolution
                resizedRegionMap = cv2.resize(croppedRegionMap, (int(outputResX),int(outputResY)))
                newRegionMapFilePath = './resizedRegionMap.png'
                cv2.imwrite(newRegionMapFilePath, resizedRegionMap)

                # crop region map with skin
                #croppedRegionMapSkin = regionmapskin[offset[1]:offset[1]+croppedHeight,offset[0]:offset[0]+croppedWidth,...].copy()
                # resize to output resolution
                #resizedRegionMapSkin = cv2.resize(croppedRegionMapSkin, (int(outputResX),int(outputResY)))
                #newRegionMapSkinFilePath = './resizedRegionMapSkin.png'
                #cv2.imwrite(newRegionMapSkinFilePath, resizedRegionMapSkin)

                #print(np.shape(croppedImage))
                #print(np.shape(croppedRegionMap))
                #print(np.shape(croppedRegionMapSkin))

                #print(np.shape(resizedImage))
                #print(np.shape(resizedRegionMap))
                #print(np.shape(resizedRegionMapSkin))

                #cv2.imshow("regionmap", regionmap)
                


                cv2.ellipse(img,pupilEllipse,(0,255,0),1)
                cv2.circle(img,(int(pupilCenter[0]),int(pupilCenter[1])),1,(0,255,0),2)
                #cv2.ellipse(img,irisEllipse,(0,255,255),1)
                #cv2.circle(img,(int(irisCenter[0]),int(irisCenter[1])),1,(0,255,255),2)
                #cv2.ellipse(img,eyeballEllipse,(255,0,255),1)
                #cv2.circle(img,(int(eyeballcenter[0]),int(eyeballcenter[1])),1,(255,0,255),2)
                
               
                pupilCenter = ((pupilCenter[0] - offset[0]), (pupilCenter[1] - offset[1]))
                pupilEllipse = (pupilCenter, pupilEllipse[1], pupilEllipse[2])
                irisCenter = ((irisCenter[0] - offset[0]), (irisCenter[1] - offset[1]))
                irisEllipse = (irisCenter, irisEllipse[1], irisEllipse[2])
                eyeballcenter = ((eyeballcenter[0] - offset[0]), (eyeballcenter[1] - offset[1]))
                eyeballEllipse = (eyeballcenter, eyeballEllipse[1], eyeballEllipse[2])


                #cv2.ellipse(croppedImage,pupilEllipse,(0,255,0),1)
                #cv2.circle(croppedImage,(int(pupilCenter[0]),int(pupilCenter[1])),1,(0,255,0),2)
                #cv2.ellipse(croppedImage,irisEllipse,(0,255,255),1)
                #cv2.circle(croppedImage,(int(irisCenter[0]),int(irisCenter[1])),1,(0,255,255),2)
                #cv2.ellipse(croppedImage,eyeballEllipse,(255,0,255),1)
                #cv2.circle(croppedImage,(int(eyeballcenter[0]),int(eyeballcenter[1])),1,(255,0,255),2)

                horScale = outputResX / float(np.shape(croppedImage)[1])
                vertScale =  outputResY / float(np.shape(croppedImage)[0])
                

                pupilCenter = (horScale * pupilCenter[0], vertScale * pupilCenter[1])
                pupilEllipse = scaleEllipseNoAngle(pupilEllipse, horScale, vertScale)
                pupilEllipse = (pupilCenter, pupilEllipse[1], pupilEllipse[2])
                
                irisCenter = (horScale * irisCenter[0], vertScale * irisCenter[1])
                irisEllipse = scaleEllipseNoAngle(irisEllipse, horScale, vertScale)
                irisEllipse = (irisCenter, irisEllipse[1], irisEllipse[2])

                eyeballcenter = (horScale * eyeballcenter[0], vertScale * eyeballcenter[1])
                eyeballEllipse = scaleEllipseNoAngle(eyeballEllipse, horScale, vertScale)
                eyeballEllipse = (eyeballcenter, eyeballEllipse[1], eyeballEllipse[2])


                #cv2.ellipse(resizedImage,pupilEllipse,(0,255,0),1)
                #cv2.circle(resizedImage,(int(pupilCenter[0]),int(pupilCenter[1])),1,(0,255,0),2)
                #cv2.ellipse(resizedImage,irisEllipse,(0,255,255),1)
                #cv2.circle(resizedImage,(int(irisCenter[0]),int(irisCenter[1])),1,(0,255,255),2)
                #cv2.ellipse(resizedImage,eyeballEllipse,(255,0,255),1)
                #cv2.circle(resizedImage,(int(eyeballcenter[0]),int(eyeballcenter[1])),1,(255,0,255),2)

                #cv2.imshow("img",img)
                #cv2.imshow("croppedImage",croppedImage)
                #cv2.imshow("resizedImage",resizedImage)
                #cv2.waitKey(0)

                #print(pupilEllipse)
                #pdb.set_trace()
                #print(img.shape)

                # add info to labels
                labels['float_pupil_x'][idx] = pupilCenter[0]
                labels['float_pupil_y'][idx] = pupilCenter[1]
                labels['float_pupilellipse_major'][idx] = pupilEllipse[1][0]
                labels['float_pupilellipse_minor'][idx] = pupilEllipse[1][1]
                labels['float_pupilellipse_angle'][idx] = pupilEllipse[2]

                labels['float_iris_x'][idx] = irisCenter[0]
                labels['float_iris_y'][idx] = irisCenter[1]
                labels['float_irisellipse_major'][idx] = irisEllipse[1][0]
                labels['float_irisellipse_minor'][idx] = irisEllipse[1][1]
                labels['float_irisellipse_angle'][idx] = irisEllipse[2]

                labels['float_eyeball_x'][idx] = eyeballcenter[0]
                labels['float_eyeball_y'][idx] = eyeballcenter[1]
                labels['float_eyeball_diameter'][idx] = eyeballEllipse[1][0]
                
                validSamples.append(idx)
                
                zw.write(newImgFilePath, os.path.basename(imgFile))
                zw.write(newRegionMapFilePath, os.path.basename(regionmaskFile))
                #zw.write(newRegionMapSkinFilePath, os.path.basename(regionmaskSkinFile))

                #if (len(validSamples) >= 100):
                #    break
        
        # reduce labels to valid samples
        for k in  labels.keys():
            labels[k] = labels[k][validSamples]

        # create output csv file
        outcsvfile = os.path.join(footagefolder, 'footage_description.csv')
        utils.writeCSVFile(labels, outcsvfile, len(validSamples))
        # write VALID labels
        # close output csv file

    