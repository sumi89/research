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
from PIL import Image
import glob

from matplotlib import pyplot as plt

    
from PIL import Image
import requests
from io import BytesIO

from urllib.parse import urlparse

from bs4 import BeautifulSoup, SoupStrainer
import requests
import datetime
#from datetime import date, datetime, timedelta

#img_files = urlparse("https://sdo.gsfc.nasa.gov/assets/img/browse/2012/09/17/")
#url = "https://sdo.gsfc.nasa.gov/assets/img/browse/2012/09/17/"



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

dd = url_w_date('2012-09-15', '2012-09-17', "https://sdo.gsfc.nasa.gov/assets/img/browse/")





# this method will take the url with date and return all the images of that url
def get_image(url):
    page = requests.get(url)    
    data = page.text
    soup = BeautifulSoup(data)
    
    img_files=[]
    for link in soup.find_all('a'):
        img_file = link.get('href')
        img_files.append(img_file)
        #print(link.get('href'))
    #print(img_files)
    
    
    size = len(img_files)-5
    
    url_img = []
    for i in range(size):
        url_ = url + img_files[5+i]
        url_img.append(url_)
        
        
    #response_all=[]
    img_all=[]
    for i in range(3):
        response = requests.get(url_img[i])
        #response_all.append(response)
        img = Image.open(BytesIO(response.content))
        img_all.append(img)
        
    return img_all

ii = []
for i in range(len(dd)):
    iii = get_image(dd[i])
    ii.append(iii)
    
for img in ii:
    img.show()

    

#url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
#
#date1 = '2012-09-15'
#date2 = '2012-09-17'
#start_date = datetime.datetime.strptime(date1, '%Y-%m-%d')
##start_date = start_date.strftime('%Y/%m/%d')
#end_date = datetime.datetime.strptime(date2, '%Y-%m-%d')
#step = datetime.timedelta(days=1)
#url_date=[]
#while start_date <= end_date:
#    #date = start_date.date()
#    ##print (start_date.date())
#    dt = start_date.date()
#    dt_f = dt.strftime('%Y/%m/%d')
#    url_d = url + dt_f + '/'
#    url_date.append(url_d)
#    start_date += step
#    
#    
#print(url_date)















    
    
