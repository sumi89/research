#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:27:20 2018

@author: sumi
"""

import cv2
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import math


# converting image to numpy array and array to image
im_ = Image.open('/Users/sumi/python/research/image.jpeg')
plt.imshow(im_)
im2arr_ = np.array(im_) # im2arr.shape: height x width x channel
plt.imshow(im2arr_)
im2arr_ = im2arr_/255
plt.imshow(im2arr_)
img = np.mean(im2arr_,axis=2)
plt.imshow(img)

#arr2im_ = Image.fromarray(im2arr_)



img = cv2.imread('/Users/sumi/python/research/image.jpeg',0)
rows,cols = img.shape

dest = np.zeros((img.shape))

####### CORRECT CODE #########
#for i in range(rows):
#    for j in range(cols):
#        xo = (8.0 * math.sin(2.0 * math.pi * i / 128.0))
#        yo = (8.0 * math.sin(2.0 * math.pi * j / 128.0))
#        ix = int(min(rows-1, max(0, int(i+xo))))
#        iy = int(min(cols-1, max(0, int(j+yo))))
#        dest[i][j] = img[ix][iy]
####### END OF CORRECT CODE ####

radius =  50
center = 112
alpha = 1.3

for i in range(rows):
#for i in range(120,121):
    if i in range(center-radius,center+radius+1):
        r = int(np.sqrt(np.abs(np.square(radius) - np.square(center - i))))
        for j in range(cols):
            if j in range(center-r,center+r+1):
                iy = int(min(center+r, max(0, int(np.abs(j-r*alpha)))))
            else:
                iy = j
            #print(r,j,iy)
            dest[i][j] = img[i][iy]
    else:
        r = 0
        for j in range(cols):
            iy = j
            #print(r,j,iy)
            dest[i][j] = img[i][iy]


#for i in range(rows):
##for i in range(2,3):
#    
#    r = int(np.sqrt(np.abs(np.square(radius) - np.square(radius - i))))
#    for j in range(112-r, 112+r+1):
#        ix = int(i)
#        iy = int(min(cols-1, max(100, int(j-r*alpha))))
#        print(j,iy)
#        dest[i][j] = img[ix][iy]


plt.imshow(dest)
plt.imshow(img)


######## image warping with openCV library #########
#cv2.imshow('img', dest)
#cv2.waitKey(0)

#
#img = cv2.imread('/Users/sumi/python/research/messi5.jpg',0)
#rows,cols = img.shape
#
#dest = np.zeros((img.shape))
#
r = 30.3
alpha = 1.3
height, width = img.shape[:2]
dst = cv2.resize(img,(height, int(width+r*alpha)), interpolation = cv2.INTER_CUBIC)

#M = np.float32([[1,0,100],[0,1,50]])
#dst = cv2.warpAffine(img,M,(cols,rows))
##
##M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
##dst = cv2.warpAffine(img,M,(cols,rows))
##
##cv2.imshow('img', img)
cv2.imshow('img',dst)
cv2.waitKey(0)
######## image warping with openCV library #########


######## vertical wave #########
img = cv2.imread('/Users/sumi/python/research/messi5.jpg', cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape

dest = np.zeros(img.shape, dtype=img.dtype)

r = 3
alpha = 360/36.5

for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        #offset_x = int(alpha * r)
        offset_y = 0
        if j+offset_x < rows:
            dest[i,j] = img[i,(j+offset_x)%cols]
#            dest[i,j] = img[i,(j+offset_x)]
        else:
            dest[i,j] = 0

#cv2.imshow('Input', img)
cv2.imshow('Vertical wave', dest)
cv2.waitKey(0)
############ end of vertical wave #########

