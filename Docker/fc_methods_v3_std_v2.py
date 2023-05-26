import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from tensorflow.keras import layers
#from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.constraints import max_norm
from tensorflow.keras.callbacks import LearningRateScheduler
import math
import os
import rclone

import tensorflow as tf

import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow
tensorflow.random.set_seed(0)
tensorflow.keras.backend.set_floatx('float64')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from numpy.random import seed
seed(0)

def split_multi_step(num_val,win_l,pred_l, start_p, end_p):#multi LSTM, multi CNN, multi ANN, # call with ts upto training time step, not necessarily
    #end_pos=len(num_val)-(win_l+pred_l)
    end_pos=(end_p-start_p)-(win_l+pred_l)

    X=[]
    y=[]
    for i in np.arange(start_p, start_p+end_pos+1):
        X.append(num_val[i:i+win_l])
        y.append(num_val[i+win_l:i+win_l+pred_l ])
    
    return np.asarray(X), np.asarray(y)

def fc_cnn(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,sample_pix_v, sample_pix_h,tile,w_dir_wt, w_dir_fc,w_dir_comp):
    win_l=60
    pred_l=1
    multi_pred_l=30
    

    verbose=1
    max_epochs=100
    batch_size = 64
    n_timesteps=X_m_tr.shape[1]
    n_features =X_m.shape[2]
    n_outputs= y_m.shape[1]
    
   
    multiStepCNN = tf.keras.Sequential([
        tf.keras.layers.Conv1D(90,9,activation='relu',strides=1, padding='same', input_shape=(n_timesteps,n_features)),
        tf.keras.layers.MaxPool1D(2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Conv1D(45,9,activation='relu',strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Conv1D(30,6,activation='relu',strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Conv1D(20,6,activation='relu',strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20, activation='relu', kernel_constraint=max_norm(3)),
        tf.keras.layers.Dense(15, activation='relu', kernel_constraint=max_norm(3)),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.00005, patience=5, verbose=1)
    multiStepCNN.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError())
    
    history = multiStepCNN.fit(
        train_inp,train_op,
        epochs=max_epochs, 
        #validation_split=0.2,
        batch_size=batch_size,
        validation_data=(val_inp, val_op),
        callbacks=[es_callback],
        shuffle=True)
    
    
    multiStepCNN.save_weights(str(Path("/app/temp_data",f"wts_multiCNN_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_v2.h5")))
    
    
    

    y_hat=multiStepCNN.predict(X_m)
    with open(str(Path("/app/temp_data",f"multiCNN_pred_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
    
    
    
def fc_ann(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,sample_pix_v, sample_pix_h,tile,w_dir_wt, w_dir_fc,w_dir_comp):
    win_l=60
    pred_l=1
    multi_pred_l=30
    

    verbose=1
    epochs=100
    batch_size = 64
    n_timesteps=X_m_tr.shape[1]
    n_features =X_m.shape[2]
    n_outputs= y_m.shape[1]
    
    X_m=np.reshape(X_m, (X_m.shape[0],X_m.shape[1]))
    y_m=np.reshape(y_m, (y_m.shape[0],y_m.shape[1]))
    train_inp=np.reshape(train_inp, (train_inp.shape[0],train_inp.shape[1]))
    train_op=np.reshape(train_op, (train_op.shape[0],train_op.shape[1]))
    val_inp=np.reshape(val_inp, (val_inp.shape[0],val_inp.shape[1]))
    val_op=np.reshape(val_op, (val_op.shape[0],val_op.shape[1]))
    

    multiStepANN = tf.keras.Sequential([
        tf.keras.layers.Dense(60,activation='relu', input_shape=(n_timesteps,), kernel_constraint=max_norm(2)),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(45,activation='relu', kernel_constraint=max_norm(3)),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(25,activation='relu', kernel_constraint=max_norm(3)),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.00005, patience=5, verbose=1)
    multiStepANN.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError(), metrics=tf.metrics.MeanAbsoluteError())
    
    
    
    history = multiStepANN.fit(train_inp,train_op, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(val_inp, val_op),
            callbacks=[es_callback],
            shuffle=True)
    
    
    

    
    
    multiStepANN.save_weights(str(Path("/app/temp_data",f"wts_multiANN_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_v2.h5")))
    
    y_hat=multiStepANN.predict(X_m)
    with open(str(Path("/app/temp_data",f"multiANN_pred_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
    
    
    
def fc_lstm_tf(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,sample_pix_v, sample_pix_h,tile,w_dir_wt, w_dir_fc,w_dir_comp):
    win_l=60
    pred_l=1
    multi_pred_l=30
    n_timesteps=X_m_tr.shape[1]
    n_features =X_m.shape[2]
    n_outputs= y_m.shape[1]
    batch_size = 64
    epoch=1

    multi_LSTM = tf.keras.Sequential([
        tf.keras.layers.LSTM(45, return_sequences=True, input_shape=(n_timesteps, n_features), activity_regularizer=regularizers.l2(1e-2), kernel_constraint=max_norm(3)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(30, activity_regularizer=regularizers.l2(1e-2), kernel_constraint=max_norm(3)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(30,activation='relu',activity_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.Dense(15,activation='relu',activity_regularizer=regularizers.l2(1e-3)),#activation='relu'
        tf.keras.layers.Dense(n_outputs)
    ])
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.00005, patience=5, verbose=1)
    multi_LSTM.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError(), metrics=tf.metrics.MeanAbsoluteError())
    
    
    history = multi_LSTM.fit(train_inp,train_op, 
            epochs=40, 
            batch_size=batch_size,
            validation_data=(val_inp, val_op),
            callbacks=[es_callback],
            shuffle=True)
            
    
    multi_LSTM.save_weights(str(Path("/app/temp_data",f"wts_multiLSTM_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_with_relu_v2.h5")))
    
    
            
    y_hat=multi_LSTM.predict(X_m)
    
    with open(str(Path("/app/temp_data",f"multiLSTM_pred_{sample_pix_v}_{sample_pix_h}_{tile}_default_lr_with_relu_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
    
