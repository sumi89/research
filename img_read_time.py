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
date1 = '2017-01-01 00:00:00'
date2 = '2017-01-03 23:59:00'
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
url_dates=[]
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

###       #select timespan        
    img_files_time = []     # image files with time
    for m in range(len(img_files_wr)):
    #for m in range(7, 8):
        #print(m)
        splitting = re.split(r'[_.?=;/]+',img_files_wr[m])
        #print('splitting', splitting)
        time_datetime = datetime.datetime.strptime(splitting[1], '%H%M%S').time()
        #print('time',time_datetime)
        
        start_date_hr = start_date + datetime.timedelta(hours = 1)    
        print('start_date_hr',start_date_hr.time())
        while (start_date_hr.time() <= end_date.time()):  
            for sec in range(-366,367):
                start_date_range = start_date_hr + datetime.timedelta(seconds = sec)
                #print(start_date_range.time())
                if (time_datetime == start_date_range.time()):
                    img_files_time.append(img_files_wr[m])
                    #print('correct')
                    #print(start_date_hr.time())
                    start_date_hr += datetime.timedelta(hours = 1)
                    #print('after',start_date_hr.time()) 
                    break
            start_date_hr += datetime.timedelta(hours = 1)
            #print('up',start_date_hr.time())
           
            if dt != start_date_hr.date():
                #print('t')
                break 
    img_files_time.append(img_files_wr[len(img_files_wr)-1])
    start_date += step            
          
                       
    size = len(img_files_time)
    url_dates_imgs = []
    for j in range(size): #range(size)
        #url_ = url_dates[i] + img_files_time[j]
        url_ = url_d + img_files_time[j]
        url_dates_imgs.append(url_)
    #urls_dates_images.append(url_dates_imgs)
    url_dates.append(url_dates_imgs)

# converting a list of list to a list
#urls_dates_images = list(chain.from_iterable(urls_dates_images))  
urls_dates_images = list(chain.from_iterable(url_dates))       
