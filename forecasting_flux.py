#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 22:15:42 2018

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

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

######### trial ################
xxxx = np.array([0, 1, 4, 9, 16, 25])
plt.plot(np.arange(len(xxxx)),xxxx)
plt.show()
######### trial ################



### flux #####
flux = np.loadtxt('/Users/sumi/python/research/changed_flux_2017/flux_2017.txt')

### normalized with mean and std ###
mean_flux = np.mean(flux)
std_flux = np.std(flux)
norm_flux = flux - mean_flux
norm_flux /= std_flux

### log of flux ###
#log_flux = np.abs(np.log(flux))
log_flux = np.log10(flux)
log_minmax_flux = (log_flux - np.min(log_flux, axis = 0))/(np.max(log_flux, axis = 0) - np.min(log_flux, axis = 0))



### normalized with min and max ###
#minmax_flux = (flux - min(flux))/(max(flux) - min(flux))
minmax_flux = (flux - np.min(flux, axis = 0))/(np.max(flux, axis = 0) - np.min(flux, axis = 0))



##### plotting different flux ######
plt.plot(np.arange(len(flux)),flux)
plt.show()

plt.plot(np.arange(len(norm_flux)),norm_flux)
plt.show()


plt.plot(np.arange(len(log_flux)),log_flux)
plt.show()

plt.plot(np.arange(len(minmax_flux)),minmax_flux)
plt.show()

#####################################################
'''' forecasting with naive approach''''
#####################################################
#def generator(target_flux, lookback, delay, min_index, max_index,
#              shuffle=False, batch_size=5, step=1):
#    if max_index is None:
#        max_index = len(data) - delay - 1
#    i = min_index + lookback
#    while 1:
##        if shuffle:
##            rows = np.random.randint(
##                min_index + lookback, max_index, size=batch_size)
##        else:
#        if i + batch_size >= max_index:
#            i = min_index + lookback
#        rows = np.arange(i, min(i + batch_size, max_index))
#        i += len(rows)
#
#        samples = np.zeros((len(rows),
#                           lookback // step,target_flux.shape[1]))
#        targets = np.zeros((len(rows),))
#        for j, row in enumerate(rows):
#            indices = range(rows[j] - lookback, rows[j], step)
#            samples[j] = target_flux[indices]
#            #print(samples[j].shape)
#            targets[j] = target_flux[rows[j] + delay]
#        #samples = samples.reshape(samples.shape[0],samples.shape[1],1)
#        #print(samples.shape)
#        yield samples, targets
        
def generator(data, lookback, delay, min_index, max_index,
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
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        

lookback = 4
step = 1
delay = 1
batch_size = 3
#images = image_dataset[0:100, :]
#target_flux = norm_flux[0:100]

train_gen = generator(log_minmax_flux,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=6500,
                      shuffle=False,
                      step=step, 
                      batch_size=batch_size)

val_gen = generator(log_minmax_flux,
                    lookback=lookback,
                    delay=delay,
                    min_index=6501,
                    max_index=7500,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(log_minmax_flux,
                     lookback=lookback,
                     delay=delay,
                     min_index=7501,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (7500 - 6501 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(minmax_flux) - 7501 - lookback) // batch_size

#def  evaluate_naive_method ():
#    batch_maes = []
#    for step in range(val_steps):
#        samples, targets = next(val_gen)
#        #print("samples.shape", samples.shape)
#        #print('samples:', samples)
#        preds = np.mean(samples)
#        #print("preds.shape",preds.shape)
#        #print("preds:", preds )
#        mae = np.mean(np.abs(preds - targets))
#        batch_maes.append(mae)
#    naive_result = np.mean(batch_maes)
#    #sss = samples[-1]
#    #ppp = preds[-1]
#    print(naive_result)
#    
#evaluate_naive_method()


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()

#####################################################
'''' forecasting with naive approach (end)''''
#####################################################


###################### basic machine learning approach ############
#model = Sequential()
#model.add(layers.Flatten(input_shape=(lookback // step, 1)))
#model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dense(1))
#model.summary()
#
#model.compile(optimizer=RMSprop(), loss='mae')
#history = model.fit_generator(train_gen,
#                              steps_per_epoch=500,
#                              epochs=5,
#                              validation_data=val_gen,
#                              validation_steps=val_steps)
####################### basic machine learning approach (end) ###############


############### recurrent baseline - GRU ##############
model = Sequential()
model.add(layers.GRU(4, dropout=0.5, recurrent_dropout=0.5, input_shape=(None, log_minmax_flux.shape[-1])))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))
model.add(layers.BatchNormalization())
model.summary()

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=1,),
                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5', save_best_only=True,)]

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=6000,
                              epochs=10,
                              validation_data=val_gen,
                              callbacks=callbacks_list,
                              validation_steps=val_steps)
import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

############### recurrent baseline -GRU (end)  ##############


############### another recurrent baseline - LSTM  ##############

from  keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


model = Sequential()
model.add(layers.LSTM(4, dropout=0.5, recurrent_dropout=0.5, input_shape=(None, log_minmax_flux.shape[-1])))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.0001), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=2000,
                              epochs=10,
                              validation_data=val_gen,
                              validation_steps=val_steps)


import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

############### another recurrent baseline - LSTM (end)  ##############


################## recurrent dropout to fight overfitting - GRU #################
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(240, 1)))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=438,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

################## recurrent dropout to fight overfitting - GRU (end) #################




################## recurrent dropout to fight overfitting - LSTM #################
model = Sequential()
model.add(layers.LSTM(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(240, 1)))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=438,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

################## recurrent dropout to fight overfitting - LSTM (end) #################



############# Stacking recurrent layers - GRU (Loss and val loss is nan )###################################
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, 1)))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=438,
                              epochs=20,
                              validation_data=val_gen,
                              callbacks=callbacks_list,
                              validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
############# Stacking recurrent layers - GRU (end) ###################################




############# Stacking recurrent layers - LSTM (Loss and val loss is nan )###################################
model = Sequential()
model.add(layers.LSTM(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(240, 1)))
model.add(layers.LSTM(64, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=438,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
############# Stacking recurrent layers - LSTM(end) ###################################


############## conv1D ###############################
model = Sequential()
model.add(layers.Conv1D(64, 5, activation='relu',
                        input_shape=(240, 1)))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=600,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
############## conv1D (end) ###############################












































