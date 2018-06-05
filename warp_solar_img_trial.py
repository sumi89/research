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
from io import BytesIO
import urllib.request
from urllib.request  import urlopen
import cv2
import PIL
from matplotlib import pyplot as plt   
from PIL import Image
import requests
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
import requests
import datetime
from itertools import chain
import math
from skimage.io import imread
from skimage.io.imread import imread_from_blob
img_data = imread_from_blob(data, 'jpg')


#this method will take start date, end date and url, return url with date
date1 = '2012-09-15'
date2 = '2012-09-16'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"


start_date = datetime.datetime.strptime(date1, '%Y-%m-%d')
#start_date = start_date.strftime('%Y/%m/%d')
end_date = datetime.datetime.strptime(date2, '%Y-%m-%d')
step = datetime.timedelta(days=1)
url_date=[]
while start_date <= end_date:
    #date = start_date.date()
    ##print (start_date.date())
    dt = start_date.date()
    dt_f = dt.strftime('%Y/%m/%d')
    url_d = url + dt_f + '/'
    url_date.append(url_d)
    start_date += step


# this method will take the url with date, return the url with date and image file name (with wavelength)
url_date_image = [] 
for i in range(len(url_date)):
    page = requests.get(url_date[i])    
    data = page.text
    soup = BeautifulSoup(data)
    # get the image file name
    img_files=[]
    for link in soup.find_all('a'):
        img_file = link.get('href')
        splitting = re.split(r'[_.?=;/]+',img_file)
        #select the wavelength
        if (splitting[3]=='4500'):
            img_files.append(img_file)
           
    size = len(img_files)
    url_date_img = []
    for i in range(1): #range(size)
        url_ = url_d + img_files[i]
        url_date_img.append(url_)
    url_date_image.append(url_date_img)

# converting a list of list to a list
url_date_image = list(chain.from_iterable(url_date_image))
        
    
# this method will take the url with date and image name, return the corresponding images 
img_all=[]
for i in range(len(url_date_image)):
    response = requests.get(url_date_image[i])
    img = Image.open(BytesIO(response.content))
    #img = np.array(img)     # converting image to a numpy array
    #img = img/255        # scaling from [0,1]
    #img = np.mean(img,axis=2) #take the mean of the R, G and B  
    img_all.append(img)   

plt.imshow(img_all[0])

img = img_all[0]
img = np.array(img) # img.shape: height x width x channel
img = img/255        # scaling from [0,1]
img = np.mean(img,axis=2) #take the mean of the R, G and B  

rows,cols = img.shape[:2]

dest = np.zeros(img.shape)

radius = 500

alpha = 360/36.5
#alpha=0
#for i in range(rows):
for i in range(600,601):
    r = int(np.sqrt(np.abs(np.square(radius) - np.square(radius - i))))
    for j in range(500-r,500+r+1):
        iy = int(min(500+r-1, max(500-r, int(np.abs(j-r*alpha)))))
        print(j,iy)
        dest[i][j] = img[i][iy]
        
print(dest[400][500])
    
plt.imshow(dest)
plt.imshow(img)












