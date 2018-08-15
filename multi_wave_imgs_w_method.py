#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:28:41 2018

@author: sumi
"""
import os, shutil

from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np

import glob
import requests
import re
import urllib.request
from urllib.request  import urlopen
import cv2 
from PIL import Image
import requests
import io
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
import datetime
from itertools import chain
import math



def get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength, resolution, url):
    required_urls = []
    start_date = datetime.datetime.strptime(date1, '%Y-%m-%d %X')
    #start_date = start_date.strftime('%Y/%m/%d')
    end_date = datetime.datetime.strptime(date2, '%Y-%m-%d %X')
    step = datetime.timedelta(days = 1)
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
            if len(diff) != 0:
                #print("diff", diff)
                index = np.argmin(diff)
                #print('index', index)
                if (index == 0):
                    index_hrs_url += index 
                    #print("index_hrs_url",index_hrs_url)
                else:
                    index_hrs_url += index + 1
                    #print("index_hrs_url", index_hrs_url)
            else:
                index = index_hrs_url
                #print("index_hrs_url", index_hrs_url)
                #print("index", index)
            start_hr = index_hrs_url + 1
            #required_urls.append(img_files_wr[index_hrs_url])  
            url_date_wave_res = url_d + img_files_wr[index_hrs_url]
            required_urls.append(url_date_wave_res) 
            
        start_date += step  
    return required_urls

date1 = '2017-01-01 00:00:00'
date2 = '2017-01-02 23:59:59'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
wavelength1 = '0131'
wavelength2 = '1600'
wavelength3 = 'HMII'
resolution = '512'

required_urls1 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength1, resolution, url)
required_urls2 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength2, resolution, url)
required_urls3 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength3, resolution, url)

image_height = 512
image_width = 512
channels = 3
img_data = np.ndarray(shape=(48, image_height, image_width, channels),
                     dtype=np.float32)
# this method will take the url with date and image name, return the corresponding images 
#img_all=[]
#j = 0
for i in range(0,48):
#for i in range(16,23):
    print('i=',i)
    response1 = requests.get(required_urls1[i])
    response2 = requests.get(required_urls2[i])
    response3 = requests.get(required_urls3[i])
    img1 = Image.open(BytesIO(response1.content))
    img2 = Image.open(BytesIO(response2.content))
    img3 = Image.open(BytesIO(response3.content))
    img1 = np.array(img1) # img.shape: height x width x channel
    img1 = img1/255        # scaling from [0,1]
    img1 = np.max(img1,axis=2) #take the mean of the R, G and B  
    img2 = np.array(img2) # img.shape: height x width x channel
    img2 = img2/255        # scaling from [0,1]
    img2 = np.max(img2,axis=2) #take the mean of the R, G and B  
    img3 = np.array(img3) # img.shape: height x width x channel
    img3 = img3/255        # scaling from [0,1]
    #img3 = np.mean(img3,axis=2) #take the mean of the R, G and B 
    multi_img = np.array((img1, img2, img3))
    img_data[i] = multi_img.T
    #j += 1


for i in range(0,50):
    print('i = ', i)
    arr = (img_data[i]*255).astype('uint8')
    img = Image.fromarray(arr)
#    img.save('/Users/sumi/python/research/multi_solar_images_trial/'+str(i)+'.jpg')
    i_str = str(i)
    if len(i_str) == 1:
        i_str = str(0) + str(0) + str(0) + i_str
    elif len(i_str) == 2:
        i_str = str(0) + str(0) + i_str
    elif len(i_str) == 3: 
        i_str = str(0) + i_str
    img.save('/Users/sumi/python/research/multi_solar_images_trial/'+i_str+'.jpg')
    
from keras import applications
from  keras.applications import VGG16, InceptionV3
import keras

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

base_model.summary()

def generator_method(img_data, minmax_flux, min_index, max_index, batch_size):
    if max_index is None:
        max_index = len(img_data) - 1
    i = min_index
    while 1:
#        if shuffle:
#            rows = np.random.randint(
#                min_index + lookback, max_index, size=batch_size)
#        else:
        if i * batch_size>= max_index:
            i = min_index
        rows = np.arange(i * batch_size, min(((i + 1) * batch_size), max_index))
        i += 1
        #i += len(rows)
        #print('i=',i)
        inputs_batch = np.zeros((len(rows),
                           img_data.shape[1], img_data.shape[2], img_data.shape[3]))
        #targets = np.zeros((len(rows), 0, data.shape[1], data.shape[2], data.shape[3]))
        labels_batch = np.zeros((len(rows),))
        for j, row in enumerate(rows):
#            indices = range(rows[j] - lookback, rows[j], step)
            #print('j=',j,'row=',row)
            inputs_batch[j] = img_data[row]
            #print("each sample", inputs_batch[j].shape)
            labels_batch[j] = flux[row][0]
        #print("batch", inputs_batch.shape)
#        i += 1
#        print('i=',i)

        yield inputs_batch, labels_batch
        

batch_size = 4
min_index = 0
max_index = 24

def extract_features(img_data, minmax_flux, min_index, max_index):
    sample_count = max_index - min_index 
    features = np.zeros(shape=(sample_count, 16, 16, 512))
    labels = np.zeros(shape=(sample_count))
    generator = generator_method(img_data, flux, min_index, max_index, batch_size)
    k = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg16_base.predict(inputs_batch)
        #print('features_batch.shape', features_batch.shape)
        features[k * batch_size : (k + 1) * batch_size] = features_batch
        #print('features.shape', features.shape)
        labels[k * batch_size : (k + 1) * batch_size] = labels_batch
        #print('labels', labels)
        k += 1
        #print('k=',k)
        if k * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(img_data, log_minmax_flux, 0, 24)
validation_features, validation_labels = extract_features(img_data, minmax_flux, 24, 30)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (24, 16 * 16 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


np.savetxt('/Users/sumi/python/research/train_features_vgg16.txt', train_features , delimiter = ',') 
np.savetxt('/Users/sumi/python/research/train_labels_vgg16.txt', train_labels , delimiter = ',') 

train_features = np.loadtxt('/Users/sumi/python/research/train_features_vgg16.txt', delimiter=",").astype(float)
train_labels = np.loadtxt('/Users/sumi/python/research/train_labels_vgg16.txt', delimiter=",").astype(float)


#train_features = np.reshape(train_features, (24, 14 * 14 * 2048))
#validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
#test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#
#np.savetxt('/Users/sumi/python/research/train_features_inceptionv3.txt', train_features , delimiter = ',') 
#np.savetxt('/Users/sumi/python/research/train_labels_inceptionv3.txt', train_labels , delimiter = ',') 
#
#train_features = np.loadtxt('/Users/sumi/python/research/train_features_inceptionv3.txt', delimiter=",").astype(float)
#train_labels = np.loadtxt('/Users/sumi/python/research/train_labels_inceptionv3.txt', delimiter=",").astype(float)



def generator_lstm(img_data, minmax_flux, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step, img_data.shape[-1]))
        #targets = np.zeros((len(rows), 0, data.shape[1], data.shape[2], data.shape[3]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = img_data[indices]
            #print("each sample", samples[j].shape)
            targets[j] = minmax_flux[rows[j] + delay]
        #print("batch", samples.shape)
        yield samples, targets

        
lookback = 3
step = 1
delay = 1
batch_size = 2


train_gen = generator_lstm(train_features, train_labels,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=24,
                      shuffle=False,
                      step=step, 
                      batch_size=batch_size)


from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import RMSprop

model = models.Sequential()

model.add(LSTM(units=128, activation='tanh', input_shape = (None, 131072)))
model.add(Dense(1))
#model.summary()

model.compile(optimizer=RMSprop(lr = 0.00001), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=12, epochs=5)













########## NO NEED ###########
#img_data_reshape = img_data.reshape((1032, 3*512*512))
#np.savetxt('/Users/sumi/python/research/multi_wave_imgs.txt', img_data_reshape , delimiter = ',') 
########### NO NEED (END) ###########










