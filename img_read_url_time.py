#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:25:42 2018

@author: sumi
"""


import os
import glob
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
date1 = '2017-02-11 00:00:00'
date2 = '2017-02-15 23:59:59'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
wavelength = '0193'
resolution = '512'
#timespan = 20

#angle = 15

start_date = datetime.datetime.strptime(date1, '%Y-%m-%d %X')
#start_date = start_date.strftime('%Y/%m/%d')
end_date = datetime.datetime.strptime(date2, '%Y-%m-%d %X')
step = datetime.timedelta(days = 1)
#step = datetime.timedelta(minutes = 15)
#urls_wr=[]
required_urls = []
while start_date <= end_date:
    #date = start_date.date()
    ##print (start_date.date())
    dt = start_date.date()
    dt_f = dt.strftime('%Y/%m/%d')
    url_d = url + dt_f + '/'
    #url_dates.append(url_d)
    #start_date += step

# this method will take the url with date, return the url with date and image file name (with wavelength)
#urls_dates_images = []
#for i in range(len(url_dates)):
#    page = requests.get(url_dates[i])    
    page = requests.get(url_d) 
    data = page.text
    soup = BeautifulSoup(data)
    # get the image file name
    img_files=[]    # image files with all info like wavelength, resolution, time
    for link in soup.find_all('a'):
        img_file = link.get('href')
        img_files.append(img_file)
        
    img_files_wr = []        # image files with all info like wavelength, resolution   
    for k in range(5, len(img_files)):
        splitting = re.split(r'[_.?=;/]+',img_files[k])
        if (splitting[3] == wavelength and splitting[2] == resolution):
            img_files_wr.append(img_files[k])
    
    hrs_url = np.zeros(len(img_files_wr))
    for time in range(len(img_files_wr)):
        url_split = re.split(r'[_.?=;/]+',img_files_wr[time])
        hr_min_sec = url_split[1]
        hrs_url[time] = float(hr_min_sec[0:2]) + float(hr_min_sec[2:4])/60 + float(hr_min_sec[4:6])/3600
    
    
    start_hr = 0
    index_hrs_url = 0
    for hours in range(0, 24):
        #print("start_hr", hours)
        diff = abs(hrs_url[start_hr:len(img_files_wr)] - hours)
        #print("diff", diff)
        index = np.argmin(diff)
        #print('index', index)
        if (index == 0):
            index_hrs_url += index 
            #print(index_hrs_url)
        else:
            index_hrs_url += index + 1
            #print("index_hrs_url", index_hrs_url)
        start_hr = index_hrs_url + 1
        
        required_urls.append(img_files_wr[index_hrs_url])        
    
    
    start_date += step            
          
                       
#    size = len(img_files_time)
#    url_imgs_wr = []
#    for j in range(size): #range(size)
#        #url_ = url_dates[i] + img_files_time[j]
#        url_ = url_d + img_files_time[j]
#        url_imgs_wr.append(url_)
#    #urls_dates_images.append(url_dates_imgs)
#    urls_wr.append(url_imgs_wr)
#    
#
## converting a list of list to a list
##urls_dates_images = list(chain.from_iterable(urls_dates_images))  
#urls_images_wr = list(chain.from_iterable(urls_wr))



   
























# this method will take the url with date and image name, return the corresponding images 
#img_all=[]
for i in range(len(urls_dates_images)):
#for i in range(16,23):
    response = requests.get(urls_dates_images[i])
    img = Image.open(BytesIO(response.content))
    img.save('/Users/sumi/python/research/solar_images_2017/'+str(i+24*58-3)+'.jpg')
#    img = np.array(img) # img.shape: height x width x channel
#    img = img/255        # scaling from [0,1]
#    img = np.mean(img,axis=2) #take the mean of the R, G and B  
#    img_all.append(img) 


######## to read .txt files   ################# 

path = '/Users/sumi/python/research/flux_2017/'

for filename in glob.glob(os.path.join(path, 'goes5min_2017_*.txt')):
    data1 = np.loadtxt(filename)
    #data1 =  np.loadtxt('/Users/sumi/python/research/goes5min_2017_12_31.txt')
    time_data1 = data1[:,3]
    short_data1 = data1[:,6]
    
    hour = 0
    flux = np.zeros(24)
    tot_short = short_data1[0]
            
    for i in range(1,data1.shape[0]):
    #for i in range(1, 13):
        
        if time_data1[i]%100 != 0:
            tot_short = tot_short + short_data1[i]
            #print("if",i, tot_short, short_data1[i])
        else:
            flux[hour] = tot_short/12
            hour += 1
            tot_short = short_data1[i]
            #print("else",i, tot_short, hour, flux[hour])
    flux[23] = tot_short/12
    #print(tot_short)
    
    os.chdir(path)
    file_name = str(data1[0][0].astype(int)) + '_' +  str(data1[0][1].astype(int)) + '_' + str(data1[0][2].astype(int))
    np.savetxt(file_name+'.txt', flux)

        
        
        #************************************#















 
