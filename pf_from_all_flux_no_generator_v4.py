import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras.optimizers import RMSprop, Adam, SGD
import  matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, LSTM, Convolution2D, Dense, Dropout, Flatten
from sklearn import preprocessing
import random
from keras.losses import mean_squared_error
import  matplotlib.pyplot as plt

def box_filt(a,ws):
    res = np.copy(a)
    res[:ws,1:] = np.cumsum(a[:ws,1:],axis=0)
    for i in range(ws,res.shape[0]):
        res[i,1:] = res[i-1,1:] + a[i,1:] - a[i-ws,1:]
    for i in range(ws):
        res[i,1:]/=(i+1)    
    res[ws:,1:] = res[ws:,1:]/ws
    return res

units = 32
batch_size = 128 #
delay = 72
lookback = 72
delay_ = 5*delay
lr = 0.001 #
step = 1
target_col = 3
target = 'p1'
initial_epochs = 2
small_set_epochs = 5
retrain_epochs = 1
init_set_size = 100000
add_set_size = 10000
window_width = 12
drop_out_frac = 0

print('target', target)

data = np.load('/Users/sumi/python/research/research_proton_flux/data/xray_pf_w_time/xray_pf_5min_v4.npy').astype('float32')
d = data[:,1:]

t=[-8,-7,0,-1,-2,-2,-2,-2]
for i in range(8):
    d[:,i] = d[:,i]-t[i]

d=np.maximum(d,0)
data[:,1:]=d
data_bf = np.log10(box_filt(np.power(10,data),window_width))
print(np.std(data,axis=0))
print(np.std(data_bf,axis=0))

prediction = np.zeros((data.shape[0],1))
print("batch size = ", batch_size, "lookback = ", lookback, 
      "learning rate = ", lr, "future = ", delay_, "minutes")
print("Target:",target_col)
d = data_bf[delay:,target_col]-data_bf[:-delay,target_col]
print("Baseline mse:",np.mean(d*d))
plt.close('all')


model = models.Sequential()
model.add(LSTM(units=units, activation='tanh', input_shape = (None, data.shape[-1])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

model.compile(optimizer=Adam(lr=lr), loss='mse')
min_index_tr = lookback
epochs=initial_epochs
k=0
for i in range(init_set_size, data.shape[0]-add_set_size, add_set_size):
    print(i)
    max_index_tr = init_set_size + k*add_set_size - delay + 1
    min_index_ts = init_set_size + k*add_set_size
    max_index_ts = init_set_size + (k+1)*add_set_size
    rows_tr = np.arange(min_index_tr, max_index_tr)
    rows_ts = np.arange(min_index_ts, max_index_ts)
    print('training rows:',min_index_tr,'to',max_index_tr)
    print('testing rows:',min_index_ts,'to',max_index_ts)
    
    samples_tr = np.zeros((len(rows_tr), lookback // step, data.shape[-1]))
    targets_tr = np.zeros((len(rows_tr),))
    for j, row in enumerate(rows_tr):
        indices_tr = range(rows_tr[j] - lookback, rows_tr[j], step)
        samples_tr[j] = data[indices_tr]
        targets_tr[j] = data_bf[rows_tr[j] + delay - 1, target_col]

    samples_ts = np.zeros((len(rows_ts), lookback // step, data.shape[-1]))
    targets_ts = np.zeros((len(rows_ts),))
    for j, row in enumerate(rows_ts):
        indices_ts = range(rows_ts[j] - lookback, rows_ts[j], step)
        samples_ts[j] = data[indices_ts]
        targets_ts[j] = data_bf[rows_ts[j] + delay-1, target_col]
    
    print('training set')
#    history = model.fit(samples_tr.reshape((-1,lookback,9,1)), targets_tr, batch_size=batch_size, epochs=epochs, 
#                        verbose=2, validation_data=(samples_ts.reshape((-1,lookback,9,1)), targets_ts)) 
    
    history = model.fit(samples_tr, targets_tr, batch_size=batch_size, epochs=initial_epochs,  
          validation_data=(samples_ts, targets_ts), verbose=2) 
    epochs = retrain_epochs
#    p = model.predict(samples_ts.reshape((-1,lookback,9,1)), batch_size = batch_size,verbose=2)
    p = model.predict(samples_ts, batch_size = batch_size,verbose=1)
    prediction[min_index_ts:max_index_ts] = p
    str_i = str(i)
    str_delay = str(delay_)   


#    plt.figure()
    plt.plot(range(len(p)), targets_ts, 'b', label='Actual Flux')
    plt.plot(range(len(p)), p, 'r', label='Predicted Flux')
    plt.title('Actual vs Predicted flux for'+str_delay+'min_'+str_i+'_'+target)
    plt.savefig('/Users/sumi/python/research/research_proton_flux/result/v4/time_series/'+
                target+'/'+str_delay+'min_'+str_i+'.jpg')
    plt.legend()
    plt.show()
    plt.close('all')

    plt.plot(targets_ts, p,'.')
    plt.xlabel('Actual Flux')
    plt.ylabel('Predicted Flux')
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.title('Actual vs Predicted flux for'+str_delay+'min_'+str_i+'_'+target)
    plt.savefig('/Users/sumi/python/research/research_proton_flux/result/v4/dot/'+
                target+'/'+str_delay+'min_'+str_i+'.jpg')
    plt.show()
    plt.close('all')
    
    
    print('validiation set')
#    model.fit(samples_ts.reshape((-1,lookback,9,1)), targets_ts, batch_size=batch_size, epochs=small_set_epochs, verbose=2) 
    model.fit(samples_ts, targets_ts, batch_size=batch_size, epochs=small_set_epochs, verbose=2) 

    MSE = np.mean(np.square(p -targets_ts.reshape(-1,1)))
    print("Mean Squared Error for small validation set:", MSE)
    pred = prediction[init_set_size:max_index_ts]
    tar = data_bf[init_set_size+delay:max_index_ts+delay, target_col].reshape(-1,1)
    MSE1 = np.mean(np.square(pred-tar))
    print("Mean Squared Error up to this point:", MSE1)
    print('training on validiation set')
    k+=1

np.save('/Users/sumi/python/research/research_proton_flux/result/prediction/prediction_lstm/'+
        target+'_'+str_delay+'.npy', prediction.astype(np.float32))
print("Mean Squared Error for all:", MSE1 )    
    



