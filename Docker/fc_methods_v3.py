import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential as kSeq
from keras.layers import Dense as kDense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D as kConv1D
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
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

def fc_cnn(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile,w_dir_wt, w_dir_fc,w_dir_comp):
    win_l=60
    pred_l=1
    multi_pred_l=30
    

    verbose=1
    epochs=90
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
    #multiStepCNN.compile(optimizer="Adam", loss="mae", metrics=["mae"])
    
    
    
    #multiStepCNN.compile(optimizer="Adam", loss="mae")
    '''write_dir_wts=os.path.join(os.getcwd(),w_dir_path,'weights')
    if not os.path.exists(write_dir_wts):#AE-det
        os.makedirs(write_dir_wts)
    else:
        ('weight dir exists, writing to it')'''
        
        
    #checkpoint_path = os.path.join(write_dir_wts,'city_wts2019_multiCNN_'+UA+'default_lr.h5')
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    '''checkpoint_path=str(Path("/app/temp_data/fua",f"city_wts2019_multiCNN_{UA}_{tile}_default_lr.h5"))

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     mode='min',
                                                     save_best_only=True,
                                                     verbose=verbose)'''

    #history=multiStepCNN.fit(X_m[0:1005], y_m[0:1005], epochs=epochs, batch_size=batch_size)
            
    #multiStepCNN.compile(optimizer='Adam', loss="mae")#tf.losses.MeanSquaredError()
    multiStepCNN.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError())
    
    history = multiStepCNN.fit(
        train_inp,train_op,
        epochs=epochs, 
        #validation_split=0.2,
        batch_size=batch_size,
        validation_data=(val_inp, val_op),
        shuffle=True)
    
    #/app/temp_data/fua",f"city_wts2019_multiCNN_{poly_id}_{tile_name}_default_lr.h5
    #multiStepCNN.save_weights(os.path.join(write_dir_wts,'wts_multiCNN_'+UA+'default_lr.h5'))
    multiStepCNN.save_weights(str(Path("/app/temp_data",f"wts_multiCNN_{UA}_{tile}_default_lr_v2.h5")))
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiCNN_{UA}_{tile}_default_lr.h5")),
                                                        f"ceph:{w_dir_wt}/"])'''
    
    '''write_dir_res=os.path.join(os.getcwd(),w_dir_path,'forecasts')
    if not os.path.exists(write_dir_res):#AE-det
        os.makedirs(write_dir_res)
    else:
        ('forecast directory exists, writing to it')'''
    #os.path.join(write_dir_res,'multiCNN_pred_'+UA+'default_lr.npy'), 'wb'

    y_hat=multiStepCNN.predict(X_m)
    with open(str(Path("/app/temp_data",f"multiCNN_pred_{UA}_{tile}_default_lr_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
    
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiCNN_pred_{UA}_{tile}_default_lr.npy")),
                                                        f"ceph:{w_dir_fc}/"])'''
    
    '''with open(str(Path("/app/temp_data/fua",f"fua_{poly_id}_{tile_name}_obs.npy")), 'wb') as f:
        np.save(f, y_hat)
    
    with open(str(Path("/app/temp_data/fua",f"fua_{poly_id}_{tile_name}_obs.npy")), 'wb') as f:
                np.save(f, ts_stack)'''

    #multiStepCNN.load_weights('gl_city_subset/city_wts2019/wts_multiCNN_t2'+UA+'default_lr.h5')


    '''mse=[]
    for j in np.arange(0,y_m.shape[0]):
        err = mean_squared_error(y_m[j, :], y_hat[j, :])
        mse.append(err)
    
    mse=np.asarray(mse)
    
    write_dir_plots=os.path.join(os.getcwd(),w_dir_path,'plots')
    if not os.path.exists(write_dir_plots):#AE-det
        os.makedirs(write_dir_plots)
    else:
        ('plot dir exists, writing to it')

    plt.figure(figsize = (12,6))
    plt.subplot(2,1,1)
    plt.title(UA)
    plt.plot(y_m[:,0], color= 'black', label= 'Data')
    plt.plot(y_hat[:,0], color= 'red', label= 'Multistep CNN') 
    plt.ylabel('Prediction (step 0)') 
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(mse)
    plt.ylabel('MSE') 
    plt.savefig(os.path.join(write_dir_plots,'plot_'+UA+'-multiCNN.png'), dpi = 180) 
    plt.close()'''
    
def fc_ann(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile,w_dir_wt, w_dir_fc,w_dir_comp):
    win_l=60
    pred_l=1
    multi_pred_l=30
    

    verbose=1
    epochs=70
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
    #multiStepANN.compile(optimizer="Adam", loss="mae", metrics=["mae"])
    multiStepANN.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError(), metrics=tf.metrics.MeanAbsoluteError())
    
    #multiStepANN.fit(X_m[0:1005], y_m[0:1005], epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    '''write_dir_wts=os.path.join(os.getcwd(),w_dir_path,'weights')
    if not os.path.exists(write_dir_wts):#AE-det
        os.makedirs(write_dir_wts)
    else:
        ('weight dir exists, writing to it')'''
    
    #checkpoint_path = os.path.join(write_dir_wts,'city_wts2019_multiANN_'+UA+'default_lr.h5')
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    '''checkpoint_path=str(Path("/app/temp_data/fua",f"city_wts2019_multiANN_{UA}_{tile}_default_lr.h5"))
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     mode='min',
                                                     save_best_only=True,
                                                     verbose=verbose)'''
    
    history = multiStepANN.fit(train_inp,train_op, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(val_inp, val_op),
            shuffle=True)
    
    
    

    #multiStepANN.save_weights(os.path.join(write_dir_wts,'wts_multiANN_'+UA+'default_lr.h5'))
    
    multiStepANN.save_weights(str(Path("/app/temp_data",f"wts_multiANN_{UA}_{tile}_default_lr_v2.h5")))
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiANN_{UA}_{tile}_default_lr.h5")),
                                                        f"ceph:{w_dir_wt}/"])'''
    
    '''write_dir_res=os.path.join(os.getcwd(),w_dir_path,'forecasts')
    if not os.path.exists(write_dir_res):#AE-det
        os.makedirs(write_dir_res)
    else:
        ('forecast dir exists, writing to it')'''
        
    y_hat=multiStepANN.predict(X_m)
    with open(str(Path("/app/temp_data",f"multiANN_pred_{UA}_{tile}_default_lr_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
    
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiANN_pred_{UA}_{tile}_default_lr.npy")),
                                                        f"ceph:{w_dir_fc}/"])'''
    


    '''mse=[]
    for j in np.arange(0,y_m.shape[0]):
        err = mean_squared_error(y_m[j, :], y_hat[j, :])
        mse.append(err)
    mse=np.asarray(mse)
    
    write_dir_plot=os.path.join(os.getcwd(),w_dir_path,'plots')
    if not os.path.exists(write_dir_plot):#AE-det
        os.makedirs(write_dir_plot)
    else:
        ('plot dir exists, writing to it')
    

    plt.figure(figsize = (12,6))
    plt.subplot(2,1,1)
    plt.title(UA)
    plt.plot(y_m[:,0], color= 'black', label= 'Data')
    plt.plot(multiStepANN.predict(X_m)[:,0], color= 'red', label= 'MultiANN') 
    plt.ylabel('Prediction (step 0)') 
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(mse)
    plt.ylabel('MSE') 
    plt.savefig(os.path.join(write_dir_plot,'plot_'+UA+'-multiANN.png'), dpi = 180) 
    plt.close()'''
    
def fc_lstm_tf(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile,w_dir_wt, w_dir_fc,w_dir_comp):
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
        #multi_LSTM.add(BatchNormalization()).astype(np.float64)
        tf.keras.layers.Dropout(0.1),
        #multi_LSTM.add(LSTM(25, activation='relu',return_sequences=True))
        #multi_LSTM.add(BatchNormalization()).astype(np.float64)
        #multi_LSTM.add(Dropout(0.1))
        tf.keras.layers.LSTM(30, activity_regularizer=regularizers.l2(1e-2), kernel_constraint=max_norm(3)),
        tf.keras.layers.Dropout(0.1),
        #model.add(BatchNormalization()).astype(np.float64)
        #model.add(LSTM(15, activation='relu'))
        #model.add(BatchNormalization()).astype(np.float64)
        #model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        #model.add(Flatten())
        tf.keras.layers.Dense(30,activation='relu',activity_regularizer=regularizers.l2(1e-3)),
        tf.keras.layers.Dense(15,activation='relu',activity_regularizer=regularizers.l2(1e-3)),#activation='relu'
        tf.keras.layers.Dense(n_outputs)
    ])
    multi_LSTM.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError(), metrics=tf.metrics.MeanAbsoluteError())
    
    
    history = multi_LSTM.fit(train_inp,train_op, 
            epochs=25, 
            batch_size=batch_size,
            validation_data=(val_inp, val_op),
            shuffle=True)
            
    '''write_dir_wts=os.path.join(os.getcwd(),w_dir_path,'weights')
    if not os.path.exists(write_dir_wts):#AE-det
        os.makedirs(write_dir_wts)
    else:
        ('weight dir exists, writing to it')'''
        
    #multiStepANN.save_weights(str(Path("/app/temp_data",f"wts_multiANN_{UA}_{tile}_default_lr.h5")))
            
    #multi_LSTM.save_weights(os.path.join(write_dir_wts,'wts_multiLSTM_'+UA+'default_lr_with_relu.h5'))
    multi_LSTM.save_weights(str(Path("/app/temp_data",f"wts_multiLSTM_{UA}_{tile}_default_lr_with_relu_v2.h5")))
    
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiLSTM_{UA}_{tile}_default_with_relu.h5")),
                                                        f"ceph:{w_dir_wt}/"])'''
    
    '''write_dir_res=os.path.join(os.getcwd(),w_dir_path,'forecasts')
    if not os.path.exists(write_dir_res):#AE-det
        os.makedirs(write_dir_res)
    else:
        ('forecast dir exists, writing to it')'''
            
    y_hat=multi_LSTM.predict(X_m)
    #with open(os.path.join(write_dir_res,'multiLSTM_pred_'+UA+'default_lr_with_relu.npy'), 'wb') as f:
    with open(str(Path("/app/temp_data",f"multiLSTM_pred_{UA}_{tile}_default_lr_with_relu_v2.npy")), 'wb') as f:
        np.save(f, y_hat)
        
    '''with open(str(Path("/app/temp_data/fua",f"multiANN_pred_{poly_id}_{tile_name}_default_lr.npy")), 'wb') as f:
        np.save(f, y_hat)'''
    
    '''rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiLSTM_pred_{UA}_{tile}_default_lr_with_relu.npy")),
                                                        f"ceph:{w_dir_fc}/"])'''

    '''mse=[]
    for j in np.arange(0,y_m.shape[0]):
        #for i in np.arange(0,y_m.shape[1]):
        err = mean_squared_error(y_m[j, :], y_hat[j, :])
        mse.append(err)
    
    write_dir_plot=os.path.join(os.getcwd(),w_dir_path,'plots')
    if not os.path.exists(write_dir_plot):#AE-det
        os.makedirs(write_dir_plot)
    else:
        ('plot dir exists, writing to it')
    
    mse=np.asarray(mse)
    plt.figure(figsize = (12,6))
    plt.subplot(2,1,1)
    plt.plot(y_m[:,0], color= 'black', label= 'Data')
    plt.plot(y_hat[:,0], color= 'red', label= 'LSTM') 
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(mse)
    plt.savefig(os.path.join(write_dir_plot,'plot_'+UA+'-multiLSTM_with_relu.png'), dpi = 180) 
    plt.close()'''
