#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:26:42 2018

@author: sumi
"""

import os, shutil
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

 
'''preprocessing images: for all the images in train_dir_solar'''
train_fnames = [os.path.join(train_dir_solar, fname) for fname in os.listdir(train_dir_solar)]

image_height = 512
image_width = 512
channels = 3
train_images = np.ndarray(shape=(len(train_fnames), image_height, image_width, channels),
                     dtype=np.float32)

i = 0
for img_path in train_fnames:
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = x.reshape(512, 512, 3)
    x = x/255.
    train_images[i] = x
    i += 1

 
'''preprocessing images: for all the images in train_dir_solar'''    

########################################################################################
 
'''preprocessing images: for all the images in original_dataset_dir'''
fnames = [os.path.join(original_dataset_dir, fname) for fname in os.listdir(original_dataset_dir)]

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
'''preprocessing images: for all the images in original_dataset_dir'''    

######################################################################################

'''renaming multiple files in a folder'''
import os
path = '/Users/sumi/python/research/solar_images_2017'
files = os.listdir(path)
files = files[1:8761]
i=0

for file in files:
    i_str = str(i)
    if len(i_str) == 1:
        i_str = str(0) + str(0) + str(0) + i_str
    elif len(i_str) == 2:
        i_str = str(0) + str(0) + i_str
    elif len(i_str) == 3: 
        i_str = str(0) + i_str
    os.rename(os.path.join(path, file), os.path.join(path, i_str+'.jpg'))
    i = i+1
'''renaming multiple files in a folder'''




































