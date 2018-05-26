#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:11:03 2018

@author: sumi
"""

import requests
from bs4 import BeautifulSoup
import re
from io import BytesIO
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



def url_with_date(date1, date2, url):
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

url_dt = url_w_date('2012-09-15', '2012-09-17', "https://sdo.gsfc.nasa.gov/assets/img/browse/")


# this method will take the url with date and return all the image file of that url
def get_image_file(url_d):
    page = requests.get(url_d)    
    data = page.text
    soup = BeautifulSoup(data)
    # get the image file name
    img_files=[]
    for link in soup.find_all('a'):
        img_file = link.get('href')
        #select the wavelength
        splitting = re.split(r'[_.?=;/]+',img_file)
        if (splitting[3]=='4500'):
            img_files.append(img_file)
#    return img_files
#
#size = len(img_files)-5         
#img_split = re.split(r'[_.]+',img_files)



# this method will take the url with date and image files, return the url with date and image file
#def url_w_date_image(url_d, img_files):    
    size = len(img_files) - 5
    url_date_img = []
    for i in range(3): #range(size)
        url_ = url_d + img_files[5+i]
        url_date_img.append(url_)
    return url_date_img
        

        
    
    
# this method will take the url with date and image name and return the corresponding images 
def get_image(url_dt_img):
    img_all=[]
    for i in range(len(url_dt_img)):
        response = requests.get(url_dt_img[i])
        img = Image.open(BytesIO(response.content))
        img_all.append(img)   
    return img_all




url_date_image = []
for i in range(len(url_dt)):
    url_dt_img = url_w_date_image(url_dt[i])
    url_date_image.append(url_dt_img)
    
url_date_image = list(chain.from_iterable(url_date_image))
    
images = get_image(url_date_image)
for img in images:
    img.show()




page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)
# get the image file name
img_files=[]
for link in soup.find_all('a'):
    img_file = link.get('href')
#    img_files.append(img_file)
    #select the wavelength
    splitting = re.split(r'[_.?=;/]+',img_file)
    if (splitting[3]=='4500'):
        img_files.append(img_file)










    
    
