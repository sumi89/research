#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:35:43 2018

@author: sumi
"""

import os, shutil

from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np

#Path to the directory where the original dataset was uncompressed
original_dataset_dir = '/Users/sumi/python/research/solar_images_2017'

##Directory where you’ll store your smaller dataset
#base_dir_solar = '/Users/sumi/python/research/base_direc_solar'
#os.mkdir(base_dir_solar)
#
##Directories for the training, validation, and test splits
#train_dir_solar = os.path.join(base_dir_solar, 'train_solar')
#os.mkdir(train_dir_solar)
#validation_dir_solar = os.path.join(base_dir_solar, 'validation_solar')
#os.mkdir(validation_dir_solar)
#test_dir_solar = os.path.join(base_dir_solar, 'test_solar')
#os.mkdir(test_dir_solar)
#
##Copies the first 6,000 images to train_dir_solar
#fnames = ['{}.jpg'.format(i) for i in range(6000)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir, fname)
#    dst = os.path.join(train_dir_solar, fname)
#    shutil.copyfile(src, dst)
#    
##Copies the next 1,500 images to validation_dir_solar
#fnames = ['{}.jpg'.format(i) for i in range(6000, 7500)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir, fname)
#    dst = os.path.join(validation_dir_solar, fname)
#    shutil.copyfile(src, dst)
#    
##Copies the rest images to test_dir_solar
#fnames = ['{}.jpg'.format(i) for i in range(7500, 8760)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir, fname)
#    dst = os.path.join(test_dir_solar, fname)
#    shutil.copyfile(src, dst)
    
## Instantiating a small convnet for solar image prediction
model = models.Sequential()
model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (512, 512, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dense(1))

model.summary()

## Configuring the model for training
model.compile(optimizer = optimizers.RMSprop(lr = 1e-4), loss = 'mse', metrics = ['mae'])


flux = np.loadtxt('/Users/sumi/python/research/flux_2017/flux_2017.txt')
train_flux = flux[0:6000]
validation_flux = flux[6000:7500]
text_flux = flux[7500:8760]


''''Image preprocessing''''
fnames = [os.path.join(original_dataset_dir, fname) for fname in os.listdir(original_dataset_dir)]
fnames = fnames[1:8761]

image_height = 512
image_width = 512
channels = 3
image_dataset = np.ndarray(shape=(len(fnames), image_height, image_width, channels),
                     dtype=np.float32)

i = 0
for img_path in fnames:
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = x.reshape(512, 512, 3)
    x = x/255.
    image_dataset[i] = x
    i += 1

train_images = image_dataset[0:6000, :, :, :]
validation_images = image_dataset[6000:7500, :, :, :]
test_images = image_dataset[7500:8760, :, :, :]
''''Image preprocessing''''



## Fitting the model using a batch generator
history = model.fit(train_images, train_flux, epochs = 100, batch_size = 100,
                              validation_data = (validation_images, validation_flux))




    
    
    
    