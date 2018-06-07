#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:11:03 2018

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


#this method will take start date, end date and url, return url with date
def url_w_date(date1, date2, url):
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
    return url_date


# this method will take the url with date, return the url with date and image file name (with wavelength)
def url_w_date_image(url_d):
    page = requests.get(url_d)    
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
    return url_date_img
        
    
# this method will take the url with date and image name, return the corresponding images 
def get_image(url_dt_img):
    img_all=[]
    for i in range(len(url_dt_img)):
        response = requests.get(url_dt_img[i])
        img = Image.open(BytesIO(response.content))
        img = np.array(img)     # converting image to a numpy array
        img = img/255        # scaling from [0,1]
        img = np.mean(img,axis=2) #take the mean of the R, G and B  
        img_all.append(img)   
    return img_all


url_dt = url_w_date('2012-09-15', '2012-09-16', "https://sdo.gsfc.nasa.gov/assets/img/browse/")


url_date_image = []
for i in range(len(url_dt)):
    url_dt_img = url_w_date_image(url_dt[i])
    url_date_image.append(url_dt_img)
    
# converting a list of list to a list
url_date_image = list(chain.from_iterable(url_date_image))
    
images = get_image(url_date_image)



## showing the images
#for img in images:
#    img.show()


######## WARPING ##########
## NO NEED ########
#def url_to_image(url):
#	# download the image, convert it to a NumPy array, and then read
#	# it into OpenCV format
#	resp = urllib.request.urlopen(url)
#	image = np.asarray(bytearray(resp.read()), dtype="uint8")
#	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#	# return the image
#	return image
#
#for url in url_date_image:
#	# download the image URL and display it
#    print ("downloading %s" % (url))
#    image = url_to_image(url)
#    print("shape{}:".format(image.shape))
#### NO NEED(end) ########
    
img = images[0]  

rows,cols = img.shape[:2]

#dest = img
dest = np.zeros((img.shape))

radius = 400
center = 500
alpha = 360/36.5
#alpha=0
        
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
    
plt.imshow(dest)
plt.imshow(img)
############## end of WARPING #################

img = images[0]  
#dest = img
dest = np.zeros(img.shape)

radius = 400
center = 500
#alpha = 360/36.5
#alpha=0
alpha = .15

for i in range(center-radius,center+radius+1):
#for i in range(120,121):
    r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - i)))))
    #for j in range(int(round(center-r+r*alpha),center+r+1)):
    for j in range(center-r,center+r+1):
        iy = int(round(min(center+r, max(0, int(round(np.abs(j-r*alpha)))))))
        #print(r,j,iy)
        dest[i][j] = img[i][iy]
plt.imshow(dest)
plt.imshow(img)
















####### another way of WARPING ##############
img = images[0]
img = cv2.imread('img')

rows,cols = img.size

dest = np.zeros(img.size)


r = 3
alpha = 360/36.5

for i in range(100,900):
    for j in range(100,900):
        #offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        r = np.sqrt(np.abs(np.square(radius) - np.square(radius - i)))
        offset_x = int(alpha * r)
        offset_y = 100
        if j+offset_x < 900:
            #dest[i,j] = img[i,(j+offset_x)%cols]
            dest[i,j] = img[i,(j+offset_x)]
        else:
            dest[i,j] = 0

#cv2.imshow('Input', img)
cv2.imshow('Vertical wave', dest)


####### end of another way of WARPING ##############


height, width = img.size[:2]
dest = cv2.resize(img,(height, int(width+r*alpha)), interpolation = cv2.INTER_CUBIC)

cv2.imshow('img',dest)
cv2.waitKey(0)



#from PIL import ImageFile
#
#def getsizes(uri):
#    # get file size *and* image size (None if not known)
#    file = urllib.request.urlopen(uri)
#    size = file.headers.get("content-length")
#    if size: size = int(size)
#    p = ImageFile.Parser()
#    while 1:
#        data = file.read(1024)
#        if not data:
#            break
#        p.feed(data)
#        if p.image:
#            return size, p.image.size
#            break
#    file.close()
#    return size, None
#
#print (getsizes("https://sdo.gsfc.nasa.gov/assets/img/browse/2012/09/15/20120915_000007_1024_4500.jpg"))
#










    
    
