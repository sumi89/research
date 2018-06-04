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

r = 30.3
alpha = 1.3

for i in range(rows):
    for j in range(cols):
        r_alpha = r * alpha
        ix = int(i)
        iy = int(min(cols-1, max(0, int(j+r_alpha))))
        dest[i][j] = img[ix][iy]

plt.imshow(dest)
plt.imshow(img)

#cv2.imshow('img', dest)
#cv2.waitKey(0)




#height, width = img.shape[:2]
#dst = cv2.resize(img,(5*height, width), interpolation = cv2.INTER_CUBIC)
#
##M = np.float32([[1,0,100],[0,1,50]])
##dst = cv2.warpAffine(img,M,(cols,rows))
#
#M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#dst = cv2.warpAffine(img,M,(cols,rows))
#
##cv2.imshow('img', img)
#cv2.imshow('img',dst)
#cv2.waitKey(0)


