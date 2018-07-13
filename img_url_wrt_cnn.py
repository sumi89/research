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
 




#train_flux = flux[0:6000]
#train_flux = np.asarray(train_flux).astype('float32')
#validation_flux = flux[6000:7500]
#validation_flux = np.asarray(validation_flux).astype('float32')
#test_flux = flux[7500:8760]
#test_flux = np.asarray(test_flux).astype('float32')

''''DATA PREPROCESSING '''''
''''Image preprocessing''''
fnames = [os.path.join(original_dataset_dir, fname) for fname in os.listdir(original_dataset_dir)]
fnames = fnames[1:8761]

image_height = 512
image_width = 512
channels = 1
image_dataset = np.ndarray(shape=(len(fnames), image_height, image_width, channels),
                     dtype=np.float32)

i = 0
for img_path in fnames:
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = x/255.
    x = np.mean(x,axis=2)
    x = x.reshape(512, 512, 1)
    image_dataset[i] = x
    i += 1

#train_images = image_dataset[0:6000, :, :, :]
#validation_images = image_dataset[6000:7500, :, :, :]
#test_images = image_dataset[7500:8760, :, :, :]
''''Image preprocessing''''

'''target processing ''''
flux = np.loadtxt('/Users/sumi/python/research/flux_2017/flux_2017.txt')

## normalized flux
mean_flux = np.mean(flux)
std_flux = np.std(flux)
norm_flux = flux - mean_flux
norm_flux /= std_flux

## log flux
log_flux = np.abs(np.log(flux))



'''target processing ''''
''''DATA PREPROCESSING '''''
   
## Instantiating a small convnet for solar image prediction
model = models.Sequential()
model.add(Conv2D(64, (5,5), activation = 'relu', input_shape = (512, 512, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(512, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.summary()

model.add(Flatten())
model.add(Dense(8192, activation = 'relu'))
model.add(Dense(1))

model.summary()

## Configuring the model for training
model.compile(optimizer = optimizers.RMSprop(lr = 1e-4), loss = 'mse', metrics = ['mae'])


#start = 0
#size_of_batch = 20
#end = start + size_of_batch
#n = (end - start) / size_of_batch
#
#for i in range (0, n+1):
#    train_images = image_dataset[size_of_batch * i : size_of_batch * (i+1), :, :, :]
#    train_flux = flux[size_of_batch * i : size_of_batch * (i+1)]
    
#def get_image_flux(start, size_of_batch, end):
#    n = int((end - start) / size_of_batch)
#    
#    #train_images_batch = np.zeros(n)
#    for i in range (0, n+1):
#        train_images = image_dataset[size_of_batch * i : size_of_batch * (i+1), :, :, :]
#        train_flux = flux[size_of_batch * i : size_of_batch * (i+1)]
#    return train_images, train_flux
    

def get_image_flux(start, size_of_batch):
    i = 0
    train_images = image_dataset[size_of_batch * i + start : size_of_batch * (i+1) + start, :, :, :]
    train_flux = flux[size_of_batch * i : size_of_batch * (i+1)]
    return train_images, train_flux

## Fitting the model using a batch generator
#train_images = image_dataset[0:300, :, :, :]
#train_flux = flux[0:300]
#train_flux = np.asarray(train_flux).astype('float32')

train_images, train_flux = get_image_flux(50, 20)




train_images = image_dataset[0:100, :, :, :]
train_flux = norm_flux[0:100]
#train_flux = log_flux[0:1000]
#train_flux = flux[0:1000]
history = model.fit(train_images, train_flux, epochs = 3, batch_size = 5)







def generator(train_images, min_index, max_index, batch_size=5):      
    i = 0     
    while 1:
        if i + batch_size >= max_index:
            i = min_index + batch_size
        rows = np.arange(i, min(i + batch_size, max_index))
        #print('rows:', rows)
        
        #i_new = i + len(rows)
        #print("i=", i, "i_new = ", i_new)
        
        #samples = np.zeros((len(rows), train_images.shape[1], train_images.shape[2], train_images.shape[3]))
        #targets = np.zeros((len(rows),))
        samples = train_images[i:i+batch_size, :, :, :]
        targets = train_flux[i:i+batch_size]
#        for j in range(i, min(i + batch_size, max_index)):
#            samples[j] = train_images[j, :, :, :]
#            print("j = ", j, ",samples[j] shape", samples[j].shape)
#            targets[j] = train_flux[j]
#            print('targets[j]', targets[j])
        i += len(rows)
        #print("i=", i)
        yield samples, targets



tr_gen = generator(train_images, min_index = 0, max_index = 10, batch_size=5)

history = model.fit_generator(tr_gen, steps_per_epoch=5, epochs=3)







