#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 00:32:39 2018

@author: sumi
"""

import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import urllib.request
from urllib.request  import urlopen
import cv2
import PIL
from matplotlib import pyplot as plt   
from PIL import Image
import requests
import io
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
import requests
import datetime
from itertools import chain
import math



#this method will take start date, end date and url, return url with date
date1 = '2012-09-15'
date2 = '2012-09-16'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
wavelength = '0335'

start_date = datetime.datetime.strptime(date1, '%Y-%m-%d')
#start_date = start_date.strftime('%Y/%m/%d')
end_date = datetime.datetime.strptime(date2, '%Y-%m-%d')
step = datetime.timedelta(days=1)
url_dates=[]
while start_date <= end_date:
    #date = start_date.date()
    ##print (start_date.date())
    dt = start_date.date()
    dt_f = dt.strftime('%Y/%m/%d')
    url_d = url + dt_f + '/'
    url_dates.append(url_d)
    start_date += step


# this method will take the url with date, return the url with date and image file name (with wavelength)
urls_dates_images = [] 
for i in range(len(url_dates)):
    page = requests.get(url_dates[i])    
    data = page.text
    soup = BeautifulSoup(data)
    # get the image file name
    img_files=[]
    for link in soup.find_all('a'):
        img_file = link.get('href')
        splitting = re.split(r'[_.?=;/]+',img_file)
        #select the wavelength
        if (splitting[3]==wavelength):
            img_files.append(img_file)
           
    size = len(img_files)
    url_dates_imgs = []
    for j in range(3): #range(size)
        url_ = url_dates[i] + img_files[j]
        url_dates_imgs.append(url_)
    urls_dates_images.append(url_dates_imgs)

# converting a list of list to a list
urls_dates_images = list(chain.from_iterable(urls_dates_images))
        


# this method will take the url with date and image name, return the corresponding images 
img_all=[]
for i in range(len(urls_dates_images)):
    response = requests.get(urls_dates_images[i])
    img = Image.open(BytesIO(response.content))
    img_all.append(img) 
#    img = np.array(img)     # converting image to a numpy array
#    img = img/255        # scaling from [0,1]
#    img = np.mean(img,axis=2) #take the mean of the R, G and B  
      


plt.imshow(img_all[0])

img = img_all[0]
img = np.array(img) # img.shape: height x width x channel
img = img/255        # scaling from [0,1]
img = np.mean(img,axis=2) #take the mean of the R, G and B  



#dest = np.zeros((180,360),dtype=np.float32)


radius = 400
center = 500
alpha = 3600/36.5
#alpha=0
#alpha = 10

dest = np.zeros(img.shape)
######## image warping (orthographic projection) ########
for i in range(center-radius,center+radius+1):
#for i in range(120,121):
    r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - i)))))
    for j in range(center-r,center+r+1):
        iy = int(round(min(center+r, max(0, int(round(np.abs(j-r*math.sin(math.radians(alpha)))))))))
        #print(r,j,iy)
        dest[i][j] = img[i][iy]
plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')
######## end of image warping (orthographic projection) ########



######## equirectangular projection (FORWARD MAPPING) WRONG ONE #############
radius = 400
center = 500
alpha = 36/36.5
#alpha = 0.1
#dest = np.zeros(img.shape)
dest = np.zeros((180,360))

for i in range(center-radius,center+radius+1):
#for i in range(120,121):
    r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - i)))))
    for j in range(center-r,center+r+1):
        lat = int(round((math.asin((center-i)/radius))*180/math.pi))
        lon = int(round((math.asin((j-center)/radius))*360/math.pi/2))
        lat = (lat+180)%180
        lon = (lon+360)%360
        #iy = int(round(min(center+r, max(0, int(round(np.abs(lon-radius*math.sin(math.radians(alpha)))))))))
        #print(r,i,j,lon,lat)
        #dest[lat][lon] = img[i][iy]
        dest[lat][lon] = img[i][j]
plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')
        
        
######## end of equirectangular projection (FORWARD MAPPING) WRONG ONE#############

#***********************************************************************************************************#

######## equirectangular projection (BACKWARD MAPPING) (without using alpha)#############

dest = np.zeros((180,360))

for i in range(-90,90):
    for j in range(-180,180):
        s_i = int(round(center - ((math.sin(math.radians(i)))*radius)))
        s_j = int(round(center - ((math.sin(math.radians(-j)))*radius)))
        #s_j_nxt = int(round(min(center+r, max(0, int(round(np.abs(s_j-radius*math.sin(math.radians(alpha)))))))))
        dest[i+90][j+180] = img[s_i][s_j]
plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')

######## equirectangular projection (BACKWARD MAPPING) (without using alpha)#############

#***********************************************************************************************************#

######## equirectangular projection (BACKWARD MAPPING)#############

alpha = 360/36.5
#alpha = 0
dest = np.zeros((180,360))

for i in range(-90,90):
#for i in range(0,1):
    for j in range(-180,180):
        s_i = int(round(center - ((math.sin(math.radians(i)))*radius)))
        r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - s_i)))))
        if r == 0:
            s_j = 500
        else:
            s_j = int(round(min(center+r, max(0,center - ((math.sin(math.radians(-j)))*radius)))))
        s_j_nxt = int(round(min(center+r, max(0, int(round(np.abs(s_j-radius*math.sin(math.radians(alpha)))))))))
        #print (s_j, s_j_nxt)
        dest[i+90][j+180] = img[s_i][s_j_nxt]
        #dest1[i+90][j+180] = img[s_i][s_j]
plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')

######## equirectangular projection (BACKWARD MAPPING) #############










