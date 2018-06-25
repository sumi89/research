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
date1 = '2012-09-15 23:00:00'
date2 = '2012-09-17 01:00:00'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
wavelength = '0193'
resolution = '1024'
#timespan = 20

#angle = 15

start_date = datetime.datetime.strptime(date1, '%Y-%m-%d %X')
#start_date = start_date.strftime('%Y/%m/%d')
end_date = datetime.datetime.strptime(date2, '%Y-%m-%d %X')
step = datetime.timedelta(days = 1)
#step = datetime.timedelta(minutes = 15)
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
    img_files=[]    # image files with all info like wavelength, resolution, time
    for link in soup.find_all('a'):
        img_file = link.get('href')
        img_files.append(img_file)

###       #select timespan        
    img_files_time = []     # image files with time
    for m in range(5, len(img_files)):
    #for m in range(7600, 7700):
        #print(m)
        splitting = re.split(r'[_.?=;/]+',img_files[m])
        time_datetime = datetime.datetime.strptime(splitting[1], '%H%M%S').time()
        #print(time_datetime)
        if (time_datetime == start_date.time()):
            img_files_time.append(img_files[m])
            start_date += datetime.timedelta(minutes = 5)
        elif (time_datetime > start_date.time()):
            start_date += datetime.timedelta(minutes = 5) 
    
    img_files_wr = []        # image files with all info like wavelength, resolution   
    for k in range(len(img_files_time)):
        splitting = re.split(r'[_.?=;/]+',img_files_time[k])
        if (splitting[2] == resolution):
            img_files_wr.append(img_files_time[k])
            
                       
    size = len(img_files_wr)
    url_dates_imgs = []
    for j in range(size): #range(size)
        url_ = url_dates[i] + img_files_wr[j]
        url_dates_imgs.append(url_)
    urls_dates_images.append(url_dates_imgs)

# converting a list of list to a list
urls_dates_images = list(chain.from_iterable(urls_dates_images))        


# this method will take the url with date and image name, return the corresponding images 
img_all=[]
for i in range(len(urls_dates_images)):
    response = requests.get(urls_dates_images[i])
    img = Image.open(BytesIO(response.content))
    img = np.array(img) # img.shape: height x width x channel
    img = img/255        # scaling from [0,1]
    img = np.mean(img,axis=2) #take the mean of the R, G and B  
    img_all.append(img) 
#    img = np.array(img)     # converting image to a numpy array
#    img = img/255        # scaling from [0,1]
#    img = np.mean(img,axis=2) #take the mean of the R, G and B  
 

#img_all=[]
#for i in range(len(url_dates_imgs)):
#    response = requests.get(url_dates_imgs[i])
#    img = Image.open(BytesIO(response.content))
#    img = np.array(img) # img.shape: height x width x channel
#    img = img/255        # scaling from [0,1]
#    img = np.mean(img,axis=2) #take the mean of the R, G and B  
#    img_all.append(img)





     

img_arr = np.dstack(img_all)    # retuns 3d array (1024, 1024, 6(# images))
img_arr = img_arr.reshape(img_arr.shape[2], -1) # reshaping

targets = np.ones(6,)











plt.imshow(img_all[0],cmap='gray')

img = img_all[0]




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

######## equirectangular projection (BACKWARD MAPPING) (NOT CORRECT) #############


radius = 400
center = 500
alpha = 360/36.5
#alpha = 0
dest = np.zeros((180,360))

for i in range(-90,90):
#for i in range(0,90):
    for j in range(-180,180):
    #for j in range(0,1):
        s_i = int(round(center + ((math.sin(math.radians(i)))*radius)))
        r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - s_i)))))
        if r == 0:
            s_j = 500
        else:
            s_j = int(round(min(center+r, max(0,center + ((math.sin(math.radians(-j)))*radius)))))
        print (s_i, s_j)
        s_j_nxt = int(round(min(center+r, max(0, int(round(np.abs(s_j-radius*math.sin(math.radians(alpha)))))))))
        dest[i+90][j+180] = img[s_i][s_j_nxt]
        #dest1[i+90][j+180] = img[s_i][s_j]
plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')

######## end of equirectangular projection (BACKWARD MAPPING) (NOT CORRECT)#############


#***********************************************************************************************************#

######## equirectangular projection (BACKWARD MAPPING) (NOT SURE) #############

radius = 405
center = 512
#alpha = 360/27
alpha = 0
dest = np.zeros((181,361))

for i in range(-90,91):
#for i in range(0,1):
    s_i = int(round(center + ((math.sin(math.radians(i)))*radius)))
    r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - s_i)))))
    for j in range(-180, 181):
    #for j in range(-90, 91):
    #for j in range(-90,-89):
        s_j = int(round(center + ((math.sin(math.radians(j)))*r)))
        #print (i, j, r, s_i, s_j)
        s_j_nxt = int(round(min(center+r, max(0, int(round(np.abs(s_j-radius*math.sin(math.radians(alpha)))))))))
        #s_j_nxt = int(round(np.abs(s_j-r*math.sin(math.radians(alpha)))))
        #print (s_i, s_j, s_j_nxt)
        #dest[i][j] = img[s_i][s_j_nxt]
        dest[i+90][j+180] = img[s_i][s_j_nxt]
        #dest[i+90][j+90] = img[s_i][s_j]
        

plt.imshow(dest, cmap='gray')
plt.imshow(img, cmap='gray')

######## end of equirectangular projection (BACKWARD MAPPING) (NOT SURE) #############



############### checking ##############################

radius = 405
center = 512
alpha = 360/27
#alpha = 0
dest = np.zeros((91, 1))

s_i = np.zeros(181).astype(int)
s_j = np.zeros(181).astype(int)


for i in range(-90, 91): #lat
#for i in range(0,1):
    s_i[i] = int(round(center + ((math.sin(math.radians(i)))*radius)))
    r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - s_i[i])))))
    for j in range(135,136): #lon
    #for j in range(0,1):
        s_j[i] = int(round(center + ((math.sin(math.radians(j)))*r)))
        #img1[i][j] =img[s_i][s_j] 
        
plt.imshow(img, cmap='gray')
plt.scatter([s_j], [s_i], c = 'r')
plt.show()

############### end of checking ##############################


####### TRIAL ###########

#
#p = [-3, -2, -1, 0, 1, 2, 3]
#q = [9, 4, 1, 0, 1, 4, 9]
#plt.plot(p,q)
#
#
#
#
#p = []
#q = 1
#p.append(q)

######### Trial ##########

#******************************************#

#dest = np.zeros((181, 361))
#for lat in range(-90, 91):
#    for lon in range(-180, 181):
#        s_i = int(round(math.cos(math.radians(lat)) * math.sin(math.radians(lon/2))))
#        s_j = int(round(math.sin(math.radians(lat))))
#        dest[i+90][j+180] = img[s_i][s_j]
#plt.imshow(dest, cmap='gray')        

#******************************************#


############## CHECKING (TRIAL) #########################

#radius = 400
#center = 500
#alpha = 360/36.5
#alpha = 0
#dest = np.zeros((181, 181))
#img1 = np.zeros((91,91))
#
#
#
#for i in range(0, 90): #lat
##for i in range(0,1):
#    s_i = int(round(center + ((math.sin(math.radians(i)))*radius)))
#    for j in range(-90,-89): #lon
#    #for j in range(0,1):
#        r = int(round(np.sqrt(np.abs(np.square(radius) - np.square(center - s_i)))))
#        s_j = int(round(center + ((math.sin(math.radians(j)))*r)))
#        #print (s_i, s_j)
#        plt.imshow(img, cmap='gray')
#        plt.scatter([s_i], [s_j], c = 'r')
#        plt.show()
#        
#        
#        print (s_i, s_j)
#        img[s_i][s_j]
#        s_j_nxt = int(round(np.abs(s_j-r*math.sin(math.radians(alpha)))))
#        #print (s_i, s_j, s_j_nxt)
#        
#        dest[i+90][j+90] = img[s_i][s_j_nxt]
#        #dest[i+90][j+90] = img[s_i][s_j]
#plt.imshow(dest, cmap='gray')
#plt.imshow(img, cmap='gray')
#
##
##im = plt.imread("/Users/sumi/python/research/df.png")
##implot = plt.imshow(im)
#
## put a blue dot at (10, 20)
#plt.scatter([10], [20])
#a_i = a_j = 10
#plt.scatter([a_i], [a_j])
## put a red dot, size 40, at 2 locations:
#plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)
#
#plt.show()

############## END OF CHECKING (TRIAL) #########################





start_date = datetime.datetime.strptime(date1, '%Y-%m-%d %X')
#start_date = start_date.strftime('%Y/%m/%d')
end_date = datetime.datetime.strptime(date2, '%Y-%m-%d %X')
step = datetime.timedelta(minutes = 15)
#url_dates=[]
while start_date <= end_date:
    #date = start_date.date()
    ##print (start_date.date())
    time = start_date.time()
    #dt_f = dt.strftime('%Y/%m/%d')
    #url_d = url + dt_f + '/'
    #url_dates.append(url_d)
    print(time)
    start_date += step





urls_dates_images = [] 
for i in range(len(url_dates)):
    page = requests.get(url_dates[i])    
    data = page.text
    soup = BeautifulSoup(data)
    # get the image file name
    img_files=[]
    img_files_test=[]
    for link in soup.find_all('a'):
        img_file = link.get('href')
        splitting = re.split(r'[_.?=;/]+',img_file)

#        #select timespan
        next_time = start_date.time()
        #print("Next Time {}".format(next_time))
        #time_in_url = splitting[1]
        #time_str_punc = time_in_url[0:2] + ':' + time_in_url[2:4] + ':' + time_in_url[4:6]
        #time_datetime = datetime.datetime.strptime(time_in_url, '%H%M%S').time()
        time_datetime = datetime.datetime.strptime(splitting[1], '%H%M%S').time()
        #print("Time in datetime {}".format(time_datetime))
        
#        #select the wavelength and resolution
        if (splitting[3]==wavelength and splitting[2] == resolution and time_datetime == next_time):
            #img_files.append(img_file)
            img_files_test.append(img_file)
#        if (splitting[3]==wavelength and splitting[2] == resolution and time_datetime == next_time):
#            img_files.append(img_file)
            start_date += step
        #print("updated start date {}".format(start_date))
#        
        
#        while start_time <= end_time:
#            time_in_url = splitting[1]
#            time_str_punc = time_in_url[0:2] + ':' + time_in_url[2:4] + ':' + time_in_url[4:6]
#            time_datetime = datetime.datetime.strptime(time_url, '%H:%M:%S').time()
#            #select the wavelength and resolution
#        if (splitting[3]==wavelength and splitting[2] == resolution):
#            img_files.append(img_file)
#            start_date += step
#                
        
    size = len(img_files)
    url_dates_imgs = []
    for j in range(size): #range(size)
        url_ = url_dates[i] + img_files[j]
        url_dates_imgs.append(url_)
    urls_dates_images.append(url_dates_imgs)

# converting a list of list to a list
urls_dates_images = list(chain.from_iterable(urls_dates_images))










img_files_wr = []        # image files with all info like wavelength, resolution   
for k in range(5, len(img_files)):
    splitting = re.split(r'[_.?=;/]+',img_files[k])
    if (splitting[2] == resolution):
        img_files_wr.append(img_files[k])
        
###       #select timespan
    #next_time = start_date.time()      
img_files_time = []
#for m in range(1620, len(img_files_wr)):
for m in range(1647, 1648):
    splitting = re.split(r'[_.?=;/]+',img_files_wr[m])
    time_datetime = datetime.datetime.strptime(splitting[1], '%H%M%S').time()

    if (time_datetime == start_date.time()):
        img_files_time.append(img_files_wr[m])
        start_date += datetime.timedelta(minutes = 5)
#################################################################################

img_files_time = []
for m in range(5, len(img_files)):
#for m in range(7600, 7700):
    #print(m)
    splitting = re.split(r'[_.?=;/]+',img_files[m])
    time_datetime = datetime.datetime.strptime(splitting[1], '%H%M%S').time()
    #print(time_datetime)
    if (time_datetime == start_date.time()):
        img_files_time.append(img_files[m])
        start_date += datetime.timedelta(minutes = 5)
    if (time_datetime > start_date.time()):
        start_date += datetime.timedelta(minutes = 5)    
    #print(start_date)













