import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import math

import tensorflow as tf
print('tf version:', tf.__version__)
import random
random.seed(0)
import os
os.environ['PYTHONHASHSEED'] = '0'
#import matplotlib.pyplot as plt
import numpy as np
from fc_methods_v3 import *
from ts_met_final_v3 import *
from time import time

def get_ra(gf_ntl,observation_count):
    window_size = 30
    
    '''for t,ntl_obs in zip(observation_count,wt_avg_at_t_wt):
        print('t, ntl', t,ntl_obs)'''
    
    ra_ntls=[]
    
    print('obs count', (observation_count))
        
    for t in np.arange(0,observation_count):
        #print('ntl at t:', wt_avg_at_t_wt[t])
        if t-int(window_size/2)<0:
            window_start=0
        else:
            window_start=t-int(window_size/2)
        if (int(window_size/2)+t)>=(observation_count):
            window_end=(observation_count-1)
        else:
            window_end=(int(window_size/2)+t)
        #print('t', t, wt_avg_at_t_wt[window_start:window_end + 1])
        window_values=gf_ntl[window_start:window_end + 1]
        
        # If the window is nothing but nans
        if np.sum(np.isnan(window_values)) == len(window_values):
            # Fill the rolling average value
            ra_ntls.append(np.nan)
        # Otherwise (at least one observation)
        else:
            # Append the mean (ignoring nans)
            ra_ntls.append(np.nanmean(window_values))
        #print('ra at t:', ra_ntls)
    
    ra_ntls=np.asarray(ra_ntls)
    print('ra ntl shape',ra_ntls.shape)
    '''plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ra_ntls)
    plt.subplot(2,1,2)
    plt.plot(gf_ntl)
    plt.show()'''
    
    return ra_ntls

#normalize data from 2017 to 2019 and scale test phase accordingly
def min_max_norm(data,start, tr_ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data[start:tr_ts])
    normalized = scaler.transform(data)
    
    return scaler, normalized

#def train_forecast(n,tile,end_d,end_m,end_y,path_obs, path_date,w_dir_wt, w_dir_fc,w_dir_comp):
def train_forecast(n,tile,end_d,end_m,end_y,path_obs, path_date,w_dir_wt, w_dir_fc,w_dir_comp):
    #df = pd.read_csv('files_read/'+n,sep='\s+',header=None)
    #path=os.path.join(r_dir_path,n)
    '''print('path:',path)
    print('path:',path)
    print('n is:',n)'''
    #UA=(n.split('_'))[1]
    UA=n
    #tile=(n.split('_'))[2]
    print('CURRENT UA, tile:', UA, tile)
    print('path3:',path_obs)
    #date_file=os.path.join(path_dates,str('fua_'+UA+'_'+tile+'_date.npy'))
    #print('date file:',date_file)
    #df = pd.read_csv(str(path),sep='\s+',header=None)
    df=np.load(path_obs)
    
    #ua_list=training_len.iloc[:,0]
    #print('UA LIST:', ua_list)
    date=np.load(path_date)
    '''print('tr 0:',training_len.iloc[:,0])
    print('tr 1:',training_len.iloc[:,1])
    print('tr 2:',training_len.iloc[:,2])
    print('tr 3:',training_len.iloc[:,3])
    print('tr 3:',training_len.loc[:,3])
    print('date:', date)
    print('begin, end',date[0], date[-1])
    #ua_val=training_len[training_len['UA'].str.equals(UA)]
    #print('len ua list',len(ua_list))
    for idx,city in enumerate(ua_list):
        print('iterating',city, idx, UA)
        if ((UA)==(city)):
            ('found', idx)
            ('found', training_len.iloc[idx,1],training_len.iloc[idx,2], training_len.iloc[idx,3])
            end_day=training_len.iloc[idx,1]
            end_month=training_len.iloc[idx,2]
            end_year=training_len.iloc[idx,3]
        else:
            print('did not match',str(UA),str(city))
    #df.to_csv(os.path.join(w_dir_path,UA+'write.csv'),header=None)'''
    
    end_day=end_d
    end_month=end_m
    end_year=end_y
    
    print('END YEAR:', end_year)
    #num_val_ntl=df[:,0]
    num_val_gf=df[:,3]
    
    
    
    yr=[]
    month=[]
    day=[]
    for dd in np.arange(0, len(date)):
        yr.append(date[dd].astype(object).year)
        month.append(date[dd].astype(object).month)
        day.append(date[dd].astype(object).day)
    
    yr=np.asarray(yr)
    month=np.asarray(month)
    day=np.asarray(day)
    
    non_nan=[]
    non_nan_yr=[]
    non_nan_month=[]
    non_nan_day=[]
    #print(len(num_val_30))
    print('year',yr)
    num_val_30=num_val_gf
    print('num_val_30',num_val_gf[0] )
    for i in np.arange(0,len(num_val_30)):
        if (~np.isnan(num_val_30[i])):
            non_nan.append(num_val_30[i])
            non_nan_yr.append(yr[i])
            non_nan_month.append(month[i])
            non_nan_day.append(day[i])
    
    #d = {'day': (non_nan_day), 'month': (non_nan_month),'year':(non_nan_yr),'non-nan-ntl':non_nan}
    #d = (non_nan_day), (non_nan_month),(non_nan_yr),non_nan
    '''d_write=pd.DataFrame(data=d)
    print(d_write)
    #np.savetxt(os.path.join(w_dir_path,UA+'non_nans.csv'),d,fmt='%10.7f', delimiter=',', newline='\n',header='ntl_pred')
    print('UA:', UA)
    d_write.to_csv(os.path.join(w_dir_path,UA+'write_3.csv'))'''
    non_nan=np.asarray(non_nan)
    non_nan_yr=np.asarray(non_nan_yr)
    
    print(yr)
    #year=df.loc[1:,3]
    #year=np.asarray(year)
    start=np.asarray(np.where(non_nan_yr==2012))#changed from yr to non-nan-yr
    #start_nonnan=np.asarray(np.where(yr==2012))
    print('start array:', start)
    print('start',start[0,0])
    #end=np.asarray(np.where(year=='2016'))
    print('end yr is:',int(end_year))
    end=np.asarray(np.where(non_nan_yr==int(end_year)))
    
    print('end array:', end)
    print('end:', end[0,-1])
    #print('start, end', start, end)
    
    start_idx=start[0,0]
    end_idx=end[0,-1]
    print('UA, end year, end', UA, end_year, end_idx, non_nan[end_idx])
    #end_idx=457 #for electrification Korhogo
    #print('shape is:', yr)
    #print('yr 1:',yr[1])
    '''years=['2012','2013','2014','2015','2016', '2017','2018','2019', '2020','2021']
    #yr.loc[yr!='a','A'].index[0]
    y_ticks=[]
    y_idx=[]
    for y in years:
        #print(y)
        y_ticks.append(y)
        #print(yr.tolist.index)
        itemindex = np.where(yr==(y))
        #print((itemindex[0][1]))
        y_idx.append(itemindex[0][1])'''
    
    '''print('y ticks:', y_ticks)
    print('y idx',y_idx)
    plt.figure(figsize=(12,7))
    plt.subplot(3,1,1)
    plt.title(UA)
    plt.plot(num_val_07,label='7-day')
    plt.xticks(ticks=y_idx,labels=y_ticks)
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(num_val_14,label='14-day')
    plt.xticks(ticks=y_idx,labels=y_ticks)
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(num_val_30,label='30-day')
    plt.xticks(ticks=y_idx,labels=y_ticks)
    plt.legend()
    plt.savefig(os.path.join(w_dir_path,UA+'_fig.png'),dpi=180)
    plt.close()'''
    #num_val=num_val_30
    num_val=non_nan
    num_val=np.reshape(num_val,(non_nan.shape[0],1))
    init_ntl=num_val
    print('num_val shape:', num_val.shape)
    ra_ntls=get_ra(num_val,num_val.shape[0])
    num_val=ra_ntls
    num_val=np.reshape(num_val,(non_nan.shape[0],1))
    '''plt.figure()
    plt.subplot(2,1,1)
    plt.title('after call')
    plt.plot(init_ntl)
    plt.subplot(2,1,2)
    plt.plot(ra_ntls)
    plt.savefig(os.path.join(w_dir_path,UA+'_gf_ra.png'), dpi=180)
    plt.close()'''
    mm_obj,norm_ts_mm=min_max_norm(num_val,start_idx,end_idx)# returns the entirely normalized ts, based on parameters 2017: 2019 (training);; UPDATE TO INDEX
    #normalized = scaler.transform(num_val5)
    inversed = mm_obj.inverse_transform(norm_ts_mm)
    
    #multistep splitting; all methods are given same forecast horizon and input window
    win_l=60
    pred_l=1
    multi_pred_l=30
    
    X_m,y_m=split_multi_step(norm_ts_mm,win_l,multi_pred_l,start_idx,len(norm_ts_mm)) # 1800 corresponds to 2017-01-01 - UPDATE TO INDEX
    print(norm_ts_mm.shape)
    
    
    X_m=X_m.reshape((X_m.shape[0],X_m.shape[1],1))
    y_m=y_m.reshape((y_m.shape[0],y_m.shape[1]))
    
    #CREATING TRAINING AND VALIDATION SPLIT FOR EACH CITY
    X_m_tr=X_m[0:(end_idx-start_idx+1),:,:] # 1005: ~3yrs of training, with a 90 day window; UPDATE TO INDEX, CHECK LENGTH (1005 vs 1095)
    y_m_tr=y_m[0:(end_idx-start_idx+1),:]
    split_idx=random.sample(range(X_m_tr.shape[0]), X_m_tr.shape[0])
    tr_frac=0.8
    val_frac=1-tr_frac
    
    train_idx=split_idx[0:int(np.floor(tr_frac*X_m_tr.shape[0]))]
    val_idx=split_idx[int(np.floor(tr_frac*X_m_tr.shape[0])):len(split_idx)]
    
    train_inp=np.zeros((len(train_idx), X_m_tr.shape[1], X_m_tr.shape[2]))
    train_op=np.zeros((len(train_idx), y_m_tr.shape[1]))
    for idx, value in enumerate(train_idx):
        train_inp[idx,:,:]=X_m_tr[value,:,:]
        train_op[idx,:]=y_m_tr[value,:]
    
    val_inp=np.zeros((len(val_idx), X_m_tr.shape[1], X_m_tr.shape[2]))
    val_op=np.zeros((len(val_idx), y_m_tr.shape[1]))
    
    for idx, value in enumerate(val_idx):
        val_inp[idx,:,:]=X_m_tr[value,:,:]
        val_op[idx,:]=y_m_tr[value,:]


        
    #CALL FORECAST METHDOS
    print('calling CNN')
    fc_cnn(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from CNN prediction')
    print('calling ANN')
    fc_ann(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from ANN prediction')
    print('calling LSTM')
    fc_lstm_tf(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from LSTM prediction')
    print('predictions completed on', UA)
    
    
    


def file_reader(dir_path):
    # get list of city names in dir_path
    f=dir_path
    file_seq_test_n=[]
    for root,subdir,files_pos in os.walk(f,topdown=False):
        print('files/UAs are:',files_pos)
    for files in sorted(files_pos):
        if not files.startswith('.'):
            #print('seq test appending:',files)
            file_seq_test_n.append(files)
    
    return file_seq_test_n
    
    #forecast_city_list(poly_id, tile, training_len, zarr_path_obs, zarr_path_date, w_dir_wt, w_dir_fc,w_dir_comp)


#def forecast_city_list(poly_id, tile,end_d,end_m,end_y,path_obs, path_date,w_dir_wt, w_dir_fc,w_dir_comp, eval_dates):
def forecast_city_list(sample_pix_v, sample_pix_h, tile, end_d,end_m,end_y, zarr_path_obs, ts, w_dir_wt, w_dir_fc,w_dir_comp):
    #gets city names in the directory
    #r_dir_path=v['read_dir']
    #w_dir_path=v['write_dir']
    #path=os.path.join(r_dir_path,'obs')
    #path_dates=os.path.join(r_dir_path,'date')
    #print('--------path------',path)
    #names=file_reader(path)
    print('calling forecast methods on:', poly_id)
    #print('FILES/ UA:', names)
    #training_len=pd.read_csv('Training_dates.csv')
    #calls forecast methods
    #for n in names:
    #print('names:',n)
    s=time()
    train_forecast(sample_pix_v, sample_pix_h, tile, end_d,end_m,end_y, zarr_path_obs, ts, w_dir_wt, w_dir_fc,w_dir_comp)
    print('TOOK:', time()-s)
    print('done training/ predicting')
    print('--------------------------')
    print('computing metrics')
    #for n in names:
    #compute_metrics(poly_id, tile,end_d,end_m,end_y, path_obs,path_date, eval_dates)
    
    
'''if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-rdir','--read_dir',help='read directory')
    parser.add_argument('-wdir','--write_dir',help='write directory')
    args=parser.parse_args()

forecast_city_list(vars(args))'''