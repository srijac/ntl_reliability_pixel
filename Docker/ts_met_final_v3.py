import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.seasonal import STL
#from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import os
import math
from metric_plots import *
from pathlib import Path
from time import time

def evaluate_fn_fp_delay(d, UA,tile, num_val, ch_summary,start_idx, end_idx,d_full,win_l,method):
    gt_seg=len(d[UA])#no. of gt seg fpr eval
    
    fn=0
    tp=0
    delay=[]
    
    gt_change_marker=np.zeros((ch_summary.shape[1],1))
    print('ua',d[UA], gt_seg)
    for t in np.arange(0,gt_seg,2):
        gt_start=d[UA][t]
        gt_end=d[UA][t+1]
        
        gt_st_index=d_full.index(gt_start)
        gt_end_index=d_full.index(gt_end)
        
        flag=0
        print('----GROUND TRUTH ------', gt_st_index, gt_end_index)
        print('----flag ------', flag)
        for gt in np.arange(gt_st_index,gt_end_index+1):
            gt_change_marker[gt,0]=1
            if (ch_summary[method,gt,0])==0:
                fn+=1
            else:
                tp+=1
                if flag==0:
                    delay.append((gt-gt_st_index))
                    flag=1
                    print('starting delay at:', gt)
            if (gt==gt_end_index) & (flag==0):
                delay.append((gt-gt_st_index+1))
                print('missed seg')
    
    print('delay is:', delay, sum(delay)/(gt_seg/2))
    
    delay_avg=sum(delay)/(gt_seg/2)
    rec=(tp)/(tp+fn)
    print('recall:', tp, fn)
    print('recall:', rec, sum(delay))
    
    pre_change_median=np.median(num_val[start_idx:end_idx])# median over training steps
    disturb_th=0.1*pre_change_median
    
    fp=0
    no_signal=np.zeros((ch_summary.shape[1],1))# time steps with no large signal; any detections here are fp; no-signals are marked as 0; unstable phases as 1--det within unstable phases are not fp
    for t in np.arange(0,ch_summary.shape[1]):
        if (np.abs(num_val[start_idx+win_l+t]-pre_change_median)>disturb_th) | (gt_change_marker[t]==1):# checks for unstable steps
            no_signal[t,0]=1
        else:
            if ch_summary[method,t,0]==1:
                fp+=1
    
    prec=(tp)/(tp+fp)
    print('precision:', prec)
    
    plt.figure(figsize=(11,7))
    plt.subplot(3,1,1)
    plt.plot(num_val[start_idx:])
    plt.subplot(3,1,2)
    plt.plot(no_signal)
    plt.subplot(3,1,3)
    plt.plot(gt_change_marker)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_gt_marker.png")), dpi=180)
    plt.close()
    
    eval_markers=np.zeros((ch_summary.shape[1],2))
    eval_markers[:,0]=no_signal[:,0]
    eval_markers[:,1]=gt_change_marker[:,0]
    
    np.savetxt(str(Path("/app/temp_data",f"fua_{UA}_{tile}_eval_markers.csv")),eval_markers,fmt='%f', delimiter=',', newline='\n',header='no_signal, gt_change')
    
    return rec, prec, delay_avg

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
    '''print('ra ntl shape',ra_ntls.shape)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ra_ntls)
    plt.subplot(2,1,2)
    plt.plot(gf_ntl)
    plt.show()'''
    
    return ra_ntls

def min_max_norm(data,start, tr_ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data[start:tr_ts])
    normalized = scaler.transform(data)
    
    return scaler, normalized
    
def file_reader(r_dir_path):
    f=r_dir_path
    file_seq_test_n=[]
    for root,subdir,files_pos in os.walk(f,topdown=False):
        print('FILES/ UA for metrics:',files_pos)
    for files in sorted(files_pos):
        if not files.startswith('.'):
            file_seq_test_n.append(files)
    
    return file_seq_test_n

def split_multi_step(num_val,win_l,pred_l, start_p, end_p):#multi LSTM, multi CNN, multi ANN, # call with ts upto training time step, not necessarily
    end_pos=(end_p-start_p)-(win_l+pred_l)
    X=[]
    y=[]
    for i in np.arange(start_p, start_p+end_pos+1):
        X.append(num_val[i:i+win_l])
        y.append(num_val[i+win_l:i+win_l+pred_l ])
    
    
    return np.asarray(X), np.asarray(y)
    
def get_sum(arr, r, c):
    sum=0
    arr_stack=[]
    while (c>-1) & (r<arr.shape[0]):
        sum+=arr[r,c]
        arr_stack.append(arr[r,c])
        r=r+1
        c=c-1
    arr_stack=np.asarray(arr_stack)
    return np.median(arr_stack)    
    
def get_mse_avg(data,pred):
    mse=[]
    print('DIMENSIONS:',data.shape, pred.shape)
    for j in np.arange(0,data.shape[0]):
        err=mean_squared_error(data[j], pred[j])
        mse.append(err)
    mse=np.asarray(mse)
    return mse
    
def get_median(pred_m, avg_pred):
    #getiing median
    count=0
    for rr in np.arange(0,pred_m.shape[0]):
        if rr==0:
            for cc in np.arange(0,pred_m.shape[1]):
                avg_pred[count,0]=get_sum(pred_m,rr,cc)
                count+=1
        else:
            avg_pred[count,0]=get_sum(pred_m,rr,cc)
            count+=1
    print('median prediction from all windows completed')
    return avg_pred
    
'''def get_ens(avg_pred_ann,avg_pred_cnn,avg_pred_lstm, weights ):
    #ANN, CNN, LSTM
    c_flag=np.zeros((avg_pred_lstm.shape[0],1))
    ens=np.zeros((avg_pred_ann.shape[0],avg_pred_ann.shape[1]))
    for i in np.arange(0,avg_pred_lstm.shape[0]): # assuming only one fails,
        #avg_pred_ens[i,0]=(0.2*avg_pred_cnn[i,0]+0.4*avg_pred_ann[i,0]+0.4*avg_pred_lstm[i,0])/1
        if ((avg_pred_lstm[i,0]>1.75) | (avg_pred_lstm[i,0]<-0.75)):
            #if (((avg_pred_ann[i,0]<1.75) & (avg_pred_ann[i,0]>-0.75))& ((avg_pred_cnn[i,0]<1.75)&(avg_pred_cnn[i,0]>-0.75))):
            ens[i,0]=(avg_pred_ann[i,0]+avg_pred_cnn[i,0])/2
            c_flag[i,0]=(2/3)
            #elif if ((avg_pred_ann[i,0]>1.75) & (avg_pred_ann[i,0]<-0.75)):
            #ens[i,0]=(avg_pred_cnn[i,0])/1
            #c_flag[i,0]=(1/3)
            #elif if ((avg_pred_cnn[i,0]>1.75) & (avg_pred_cnn[i,0]<-0.75)):
            #ens[i,0]=(avg_pred_ann[i,0])/1
            #c_flag[i,0]=(1/3)
        elif ((avg_pred_cnn[i,0]>1.75) | (avg_pred_cnn[i,0]<-0.75)):
            #if (((avg_pred_ann[i,0]<1.75) & (avg_pred_ann[i,0]>-0.75))& ((avg_pred_lstm[i,0]<1.75)&(avg_pred_lstm[i,0]>-0.75))):
            ens[i,0]=(avg_pred_ann[i,0])/1
            c_flag[i,0]=(2/3)
        elif ((avg_pred_ann[i,0]>1.75) | (avg_pred_ann[i,0]<-0.75)):
            ens[i,0]=(avg_pred_cnn[i,0])/1
            c_flag[i,0]=(2/3)
        elif ((avg_pred_lstm[i,0]<1.75) & (avg_pred_lstm[i,0]>-0.75)) & ((avg_pred_cnn[i,0]<1.75) & (avg_pred_cnn[i,0]>-0.75)) & ((avg_pred_ann[i,0]<1.75) | (avg_pred_ann[i,0]>-0.75)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1
            c_flag[i,0]=1
    return ens, c_flag'''
    
'''def get_ens(avg_pred_ann,avg_pred_cnn,avg_pred_lstm, weights ):
    #ANN, CNN, LSTM
    c_flag=np.zeros((avg_pred_lstm.shape[0],1))
    ens=np.zeros((avg_pred_ann.shape[0],avg_pred_ann.shape[1]))
    for i in np.arange(0,avg_pred_lstm.shape[0]): # assuming only one fails,
        #avg_pred_ens[i,0]=(0.2*avg_pred_cnn[i,0]+0.4*avg_pred_ann[i,0]+0.4*avg_pred_lstm[i,0])/1
        if ((avg_pred_lstm[i,0]>1.75) | (avg_pred_lstm[i,0]<-0.75)):
            ens[i,0]=(avg_pred_ann[i,0]+avg_pred_cnn[i,0])/2
            c_flag[i,0]=(2/3)
        elif ((avg_pred_cnn[i,0]>1.75) | (avg_pred_cnn[i,0]<-0.75)):
            ens[i,0]=(avg_pred_ann[i,0])/1
            c_flag[i,0]=(2/3)
        elif ((avg_pred_ann[i,0]>1.75) | (avg_pred_ann[i,0]<-0.75)):
            ens[i,0]=(avg_pred_cnn[i,0])/1
            c_flag[i,0]=(2/3)
        elif ((avg_pred_lstm[i,0]<1.75) & (avg_pred_lstm[i,0]>-0.75)) & ((avg_pred_cnn[i,0]<1.75) & (avg_pred_cnn[i,0]>-0.75)) & ((avg_pred_ann[i,0]<1.75) | (avg_pred_ann[i,0]>-0.75)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1
            c_flag[i,0]=1
    return ens, c_flag'''
    
def get_ens(avg_pred_ann,avg_pred_cnn,avg_pred_lstm, weights ):
    #ANN, CNN, LSTM
    c_flag=np.zeros((avg_pred_lstm.shape[0],1))
    ens=np.zeros((avg_pred_ann.shape[0],avg_pred_ann.shape[1]))
    for i in np.arange(0,avg_pred_lstm.shape[0]): # assuming only one fails,
        #avg_pred_ens[i,0]=(0.2*avg_pred_cnn[i,0]+0.4*avg_pred_ann[i,0]+0.4*avg_pred_lstm[i,0])/1
        ann_val=0#0 invalid, 1 valid
        cnn_val=0
        lstm_val=0
        if ((avg_pred_lstm[i,0]<2) & (avg_pred_lstm[i,0]>-1)):
            lstm_val=1
        if ((avg_pred_cnn[i,0]<2) & (avg_pred_cnn[i,0]>-1)):
            cnn_val=1
        if ((avg_pred_ann[i,0]<2) & (avg_pred_ann[i,0]>-1)):
            ann_val=1
        if ((lstm_val==1) & (ann_val==1)& (cnn_val==1)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1 # den =1 because weighted sum=1
            c_flag[i,0]=1
        elif ((lstm_val==1) & (ann_val==1)& (cnn_val==0)):
            ens[i,0]=(avg_pred_ann[i,0]+avg_pred_lstm[i,0])/2
            c_flag[i,0]=2/3
        elif ((lstm_val==1) & (ann_val==0)& (cnn_val==1)):
            ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_lstm[i,0])/2
            c_flag[i,0]=2/3
        elif ((lstm_val==1) & (ann_val==0)& (cnn_val==0)):
            ens[i,0]=(avg_pred_lstm[i,0])
            c_flag[i,0]=1/3
        elif ((lstm_val==0) & (ann_val==1)& (cnn_val==1)):
            ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_ann[i,0])/2
            c_flag[i,0]=2/3
        elif ((lstm_val==0) & (ann_val==1)& (cnn_val==0)):
            ens[i,0]=(avg_pred_ann[i,0])/1
            c_flag[i,0]=1/3
        elif ((lstm_val==0) & (ann_val==0)& (cnn_val==1)):
            ens[i,0]=(avg_pred_cnn[i,0])/1
            c_flag[i,0]=1/3
        elif ((lstm_val==0) & (ann_val==0)& (cnn_val==0)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1
            c_flag[i,0]=-1
    return ens, c_flag
    
def get_ens_all_wt(avg_pred_ann,avg_pred_cnn,avg_pred_lstm, weights ):
    #ANN, CNN, LSTM
    c_flag=np.zeros((avg_pred_lstm.shape[0],1))
    ens=np.zeros((avg_pred_ann.shape[0],avg_pred_ann.shape[1]))
    for i in np.arange(0,avg_pred_lstm.shape[0]): # assuming only one fails,
        #avg_pred_ens[i,0]=(0.2*avg_pred_cnn[i,0]+0.4*avg_pred_ann[i,0]+0.4*avg_pred_lstm[i,0])/1
        ann_val=0#0 invalid, 1 valid
        cnn_val=0
        lstm_val=0
        if ((avg_pred_lstm[i,0]<2) & (avg_pred_lstm[i,0]>-1)):
            lstm_val=1
        if ((avg_pred_cnn[i,0]<2) & (avg_pred_cnn[i,0]>-1)):
            cnn_val=1
        if ((avg_pred_ann[i,0]<2) & (avg_pred_ann[i,0]>-1)):
            ann_val=1
        if ((lstm_val==1) & (ann_val==1)& (cnn_val==1)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1 # den =1 because weighted sum=1
            c_flag[i,0]=1
        elif ((lstm_val==1) & (ann_val==1)& (cnn_val==0)):
            if avg_pred_cnn[i,0]<-1:
                ens[i,0]=(avg_pred_ann[i,0]+avg_pred_lstm[i,0]-1)/3
                c_flag[i,0]=2/3
            elif avg_pred_cnn[i,0]>2:
                ens[i,0]=(avg_pred_ann[i,0]+avg_pred_lstm[i,0]+1)/3
                c_flag[i,0]=2/3
        elif ((lstm_val==1) & (ann_val==0)& (cnn_val==1)):
            if avg_pred_ann[i,0]<-1:
                ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_lstm[i,0]-1)/3
                c_flag[i,0]=2/3
            elif avg_pred_ann[i,0]>2:
                ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_lstm[i,0]+1)/3
                c_flag[i,0]=2/3
        elif ((lstm_val==1) & (ann_val==0)& (cnn_val==0)):
            if (avg_pred_ann[i,0]<-1) & (avg_pred_cnn[i,0]<-1) :
                ens[i,0]=(avg_pred_lstm[i,0]-1-1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_ann[i,0]<-1) & (avg_pred_cnn[i,0]>2) :
                ens[i,0]=(avg_pred_lstm[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_ann[i,0]>2) & (avg_pred_cnn[i,0]<-1) :
                ens[i,0]=(avg_pred_lstm[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_ann[i,0]>2) & (avg_pred_cnn[i,0]>2) :
                ens[i,0]=(avg_pred_lstm[i,0]+1+1)/3
                c_flag[i,0]=1/3  
        elif ((lstm_val==0) & (ann_val==1)& (cnn_val==1)):
            if avg_pred_lstm[i,0]<-1:
                ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_ann[i,0]-1)/3
                c_flag[i,0]=2/3
            elif avg_pred_lstm[i,0]>2:
                ens[i,0]=(avg_pred_cnn[i,0]+avg_pred_ann[i,0]+1)/3
                c_flag[i,0]=2/3
        elif ((lstm_val==0) & (ann_val==1)& (cnn_val==0)):
            if (avg_pred_lstm[i,0]<-1) & (avg_pred_cnn[i,0]<-1) :
                ens[i,0]=(avg_pred_ann[i,0]-1-1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]<-1) & (avg_pred_cnn[i,0]>2) :
                ens[i,0]=(avg_pred_ann[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]>2) & (avg_pred_cnn[i,0]<-1) :
                ens[i,0]=(avg_pred_ann[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]>2) & (avg_pred_cnn[i,0]>2) :
                ens[i,0]=(avg_pred_ann[i,0]+1+1)/3
                c_flag[i,0]=1/3  
        elif ((lstm_val==0) & (ann_val==0)& (cnn_val==1)):
            if (avg_pred_lstm[i,0]<-1) & (avg_pred_ann[i,0]<-1) :
                ens[i,0]=(avg_pred_cnn[i,0]-1-1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]<-1) & (avg_pred_ann[i,0]>2) :
                ens[i,0]=(avg_pred_cnn[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]>2) & (avg_pred_ann[i,0]<-1) :
                ens[i,0]=(avg_pred_cnn[i,0]-1+1)/3
                c_flag[i,0]=1/3
            elif (avg_pred_lstm[i,0]>2) & (avg_pred_ann[i,0]>2) :
                ens[i,0]=(avg_pred_cnn[i,0]+1+1)/3
                c_flag[i,0]=1/3  
        elif ((lstm_val==0) & (ann_val==0)& (cnn_val==0)):
            ens[i,0]=(weights[1]*avg_pred_cnn[i,0]+weights[0]*avg_pred_ann[i,0]+weights[2]*avg_pred_lstm[i,0])/1
            c_flag[i,0]=-1
    return ens, c_flag
    
def ch_summary_methods(bin_ann_avg,bin_cnn_avg, bin_lstm_avg,bin_ens_avg, avg_pred_ann,avg_pred_cnn,avg_pred_lstm,avg_pred_ens,start_idx,norm_ts_mm, win_l):
    #----------------change summary for each method separately (if change point, degree/ magnitude, direction (obs-pred))------
    ch_summary=np.zeros((4,bin_lstm_avg.shape[0],3))# models(ann,cnn,ensx lstm) x time-steps x (ch pt, degree, direction)
    for i in np.arange(0,(bin_lstm_avg.shape[0])):
        if bin_ann_avg[i,0]==1:
            ch_summary[0,i,0]=1
            ch_summary[0,i,1]=np.abs(avg_pred_ann[i]-norm_ts_mm[win_l+i+start_idx,0])# change to/ add % decline
            ch_summary[0,i,2]=-(avg_pred_ann[i]-norm_ts_mm[win_l+i+start_idx])
        if bin_cnn_avg[i,0]==1:
            ch_summary[1,i,0]=1
            ch_summary[1,i,1]=np.abs(avg_pred_cnn[i]-norm_ts_mm[win_l+i+start_idx,0])
            ch_summary[1,i,2]=-(avg_pred_cnn[i]-norm_ts_mm[win_l+i+start_idx])
        if bin_ens_avg[i,0]==1:
            ch_summary[2,i,0]=1
            ch_summary[2,i,1]=np.abs(avg_pred_ens[i]-norm_ts_mm[win_l+i+start_idx,0])
            ch_summary[2,i,2]=-(avg_pred_ens[i]-norm_ts_mm[win_l+i+start_idx])
        if bin_lstm_avg[i,0]==1:
            ch_summary[3,i,0]=1
            ch_summary[3,i,1]=np.abs(avg_pred_lstm[i]-norm_ts_mm[win_l+i+start_idx,0])
            ch_summary[3,i,2]=-(avg_pred_lstm[i]-norm_ts_mm[win_l+i+start_idx])
            
    return ch_summary

def compute_metrics(n, tile,end_d,end_m,end_y, path_obs,path_date, eval_dates):    
    #names=file_reader(r_dir_path)
    #print('FILES:', names)
    
    #df = pd.read_csv(os.path.join(r_dir_path,n),sep='\s+',header=None)
    s=time()
    UA=n
    #tile=(n.split('_'))[2]
    print('CURRENT UA, tile:', UA, tile)
    print('path3:',path_obs)
    
    df=np.load(path_obs)
    
    
    date=np.load(path_date)
    end_day=end_d
    end_month=end_m
    end_year=end_y
    
    print('END YEAR:', end_year)
    #num_val_ntl=df[:,0]
    num_val_gf=df[:,3]
    gf_flag=df[:,4]
    
    
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
    non_nan_flag=[]
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
            non_nan_flag.append(gf_flag[i])
            
    non_nan=np.asarray(non_nan)
    non_nan_yr=np.asarray(non_nan_yr)
    non_nan_month=np.asarray(non_nan_month)
    non_nan_day=np.asarray(non_nan_day)
    non_nan_flag=np.asarray(non_nan_flag)
    non_nan_d_list=non_nan_day
    non_nan_m_list=non_nan_month
    non_nan_y_list=non_nan_yr
    
    #print(yr)
    
    start=np.asarray(np.where(non_nan_yr==2012))#changed from yr to non-nan-yr
    
    #print('start array:', start)
    #print('start',start[0,0])
    #end=np.asarray(np.where(year=='2016'))
    print('end yr is:',int(end_year))
    end=np.asarray(np.where(non_nan_yr==int(end_year)))
    
    print('end array:', end)
    print('end:', end[0,-1])
    #print('start, end', start, end)
    
    start_idx=start[0,0]
    end_idx=end[0,-1]
    print('UA, end year, end', UA, end_year, end_idx, non_nan[end_idx])
    
    d_full=[]
    win_l=60
    print('list len:', len(non_nan_d_list))
    for i in np.arange(start_idx+win_l,len(non_nan_d_list)):
        #print(i,non_nan_d_list[i], non_nan_m_list[i], non_nan_y_list[i])
        #print(i,str(non_nan_d_list[i])+'-'+str(non_nan_m_list[i])+'-'+str(non_nan_y_list[i]))
        d_full.append(str(non_nan_y_list[i])+'-'+str(non_nan_m_list[i])+'-'+str(non_nan_d_list[i]))
    
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
    plt.show()'''
    print('num_val shape:', num_val.shape)
    mm_obj,norm_ts_mm=min_max_norm(num_val,start_idx,end_idx)# returns the entirely normalized ts, based on parameters 2017: 2019 (training);; UPDATE TO INDEX
    #normalized = scaler.transform(num_val5)
    #inversed = mm_obj.inverse_transform(norm_ts_mm)
    
    #print(len(num_val_30))
    
    
    #years_tick=['2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
    years_tick=[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021, 2022]
    years_tick_pos=[]
    for idx,y in enumerate(years_tick):
        print('index, yr:',idx,y)
        years_tick_pos.append(((np.asarray(np.where(non_nan_yr==y)))[0,0]-start_idx))
    years_tick_pos=np.asarray(years_tick_pos)
    
    
    #mm_obj,norm_ts_mm=min_max_norm(num_val,start_idx,end_idx)# returns the entirely normalized ts, based on parameters upto 305
    
    
    
    win_l=60
    multi_pred_l=30
    X_m,y_m=split_multi_step(norm_ts_mm,win_l,multi_pred_l,start_idx,len(norm_ts_mm))
    
    #read predictions
    
    '''pred_m_cnn=np.load(os.path.join(os.getcwd(),w_dir_path,'forecasts','multiCNN_pred_'+UA+'default_lr.npy'))
    pred_m_ann=np.load(os.path.join(os.getcwd(),w_dir_path,'forecasts','multiANN_pred_'+UA+'default_lr.npy'))
    pred_m_lstm=np.load(os.path.join(os.getcwd(),w_dir_path,'forecasts','multiLSTM_pred_'+UA+'default_lr_with_relu.npy'))'''
    
    pred_m_cnn=np.load(str(Path("/app/temp_data",f"multiCNN_pred_{UA}_{tile}_default_lr_v2.npy")))
    pred_m_ann=np.load(str(Path("/app/temp_data",f"multiANN_pred_{UA}_{tile}_default_lr_v2.npy")))
    pred_m_lstm=np.load(str(Path("/app/temp_data",f"multiLSTM_pred_{UA}_{tile}_default_lr_with_relu_v2.npy")))
    

    #get median predictions from all windows a time-step appears in
    
    avg_pred_ann=np.zeros((pred_m_ann.shape[1]+pred_m_ann.shape[0]-1,1))
    avg_pred_cnn=np.zeros((pred_m_cnn.shape[1]+pred_m_cnn.shape[0]-1,1))
    avg_pred_ens=np.zeros((pred_m_ann.shape[1]+pred_m_ann.shape[0]-1,1))
    #avg_pred_lstm1=np.zeros((pred_m_lstm1.shape[1]+pred_m_lstm1.shape[0]-1,1))
    avg_pred_lstm=np.zeros((pred_m_lstm.shape[1]+pred_m_lstm.shape[0]-1,1))
    
    avg_pred_ann=get_median(pred_m_ann,avg_pred_ann)
    avg_pred_cnn=get_median(pred_m_cnn,avg_pred_cnn)
    avg_pred_lstm=get_median(pred_m_lstm,avg_pred_lstm)
    weights=[0.3,0.2,0.5]#ann cnn, lstm
    
    avg_pred_ens, c_flag=get_ens(avg_pred_ann,avg_pred_cnn,avg_pred_lstm, weights)
    
    '''write_dir_plots=os.path.join(os.getcwd(),w_dir_path,'plots')
    if not os.path.exists(write_dir_plots):#AE-det
        os.makedirs(write_dir_plots)
    else:
        ('plot dir exists, writing to it')'''
    
    
    #----------------mse of median pred and original ntl------
    print('DIMENSIONS:',avg_pred_ann.shape,norm_ts_mm[win_l+start_idx:].shape)
    avg_mse_ann=get_mse_avg(avg_pred_ann,norm_ts_mm[win_l+start_idx:])
    avg_mse_cnn=get_mse_avg(avg_pred_cnn,norm_ts_mm[win_l+start_idx:])
    avg_mse_ens=get_mse_avg(avg_pred_ens,norm_ts_mm[win_l+start_idx:])
    #avg_mse_lstm1=get_mse_avg(avg_pred_lstm1,norm_ts_mm[win_l+1800:])
    avg_mse_lstm=get_mse_avg(avg_pred_lstm,norm_ts_mm[win_l+start_idx:])
    
    print('shapes:', norm_ts_mm[win_l+start_idx:,0].shape, avg_pred_ann.shape, avg_mse_ann.shape)
    
    observation_count=avg_mse_lstm.shape[0]
    ts_stack=np.zeros((observation_count,14))#ntl,ntl-filtered,non_weighted_avg, wt_ntl_filtered, gap-filled, gap-filled-filtered,non_weighted_gap_ wt-gap-filled, flags
    #ts_stack[:,0]=ntl[]
    #ts_stack[:,1]=filterd_ntl
    ts_stack[:,0]=non_nan_yr[start_idx+win_l:]
    ts_stack[:,1]=non_nan_month[start_idx+win_l:]
    ts_stack[:,2]=non_nan_day[start_idx+win_l:]
    ts_stack[:,3]=norm_ts_mm[win_l+start_idx:,0]
    ts_stack[:,4]=avg_pred_ann[:,0]
    ts_stack[:,5]=avg_pred_cnn[:,0]
    ts_stack[:,6]=avg_pred_lstm[:,0]
    ts_stack[:,7]=avg_pred_ens[:,0]
    ts_stack[:,8]=avg_mse_ann
    ts_stack[:,9]=avg_mse_cnn
    ts_stack[:,10]=avg_mse_lstm
    ts_stack[:,11]=avg_mse_ens
    ts_stack[:,12]=c_flag[:,0]
    ts_stack[:,13]=non_nan_flag[start_idx+win_l:]
    #ts_stack[:,6]=np.datetime64(poly_zarr["Dates"][0:])
    
    np.savetxt(str(Path("/app/temp_data",f"fua_{UA}_{tile}_pred_mse.csv")),ts_stack,fmt='%f', delimiter=',', newline='\n',header='yr, month, day, ntl_avg, pred_ann, pred_cnn, pred_lstm, pred_ens, mse_ann, mse_cnn, mse_lstm, mse_ens, ens_flag, qa_flag')
    
    
    pred_plots(avg_pred_ann, avg_pred_cnn, avg_pred_ens, avg_pred_lstm,norm_ts_mm, win_l,start_idx,years_tick, years_tick_pos, UA,tile, avg_mse_ens)
    
    
    #----------------top k mse from each method extracting threshold values------
    mse_top_ann_avg=np.sort(avg_mse_ann)[::-1]
    mse_top_cnn_avg=np.sort(avg_mse_cnn)[::-1]
    mse_top_ens_avg=np.sort(avg_mse_ens)[::-1]
    mse_top_lstm_avg=np.sort(avg_mse_lstm)[::-1]
    top_k_percent=[.05,0.08,.1,.12,.15,.2,.25]# ens should use higher threshold
    l_ann = [x * len(avg_mse_ann) for x in top_k_percent]
    l_cnn = [x * len(avg_mse_cnn) for x in top_k_percent]
    l_ens = [x * len(avg_mse_ens) for x in top_k_percent]
    l_lstm = [x * len(avg_mse_lstm) for x in top_k_percent]
    
    top_k_val_ann_avg=[mse_top_ann_avg[int(np.ceil(x))] for x in l_ann]
    top_k_val_cnn_avg=[mse_top_cnn_avg[int(np.ceil(x))] for x in l_cnn]
    top_k_val_ens_avg=[mse_top_ens_avg[int(np.ceil(x))] for x in l_ens]
    top_k_val_lstm_avg=[mse_top_lstm_avg[int(np.ceil(x))] for x in l_lstm]

    
    cols=['red','blue','green','magenta','orange','lime','coral']

    #plot_threshold(avg_mse_ann, avg_mse_cnn, avg_mse_ens, avg_mse_lstm, years_tick, years_tick_pos,write_dir_plots, UA, top_k_val_ann_avg, top_k_val_cnn_avg, top_k_val_ens_avg,top_k_val_lstm_avg, cols,top_k_percent)
    plot_threshold(avg_mse_ann, avg_mse_cnn, avg_mse_ens, avg_mse_lstm, years_tick, years_tick_pos, UA,tile, top_k_val_ann_avg, top_k_val_cnn_avg, top_k_val_ens_avg,top_k_val_lstm_avg, cols,top_k_percent)
    
    
    
    #----------------binary indicator of change vs no change for each method separately------
    ann_th_avg=top_k_val_ann_avg[6]#top 25%
    cnn_th_avg=top_k_val_cnn_avg[6]
    ens_th_avg=top_k_val_ens_avg[6]
    lstm_th_avg=top_k_val_lstm_avg[6]
    bin_ann_avg=np.zeros((avg_mse_ann.shape[0],1))# storing binary change decision using ann
    bin_cnn_avg=np.zeros((avg_mse_cnn.shape[0],1))# storing binary change decision using cnn
    bin_ens_avg=np.zeros((avg_mse_ens.shape[0],1))# storing binary change decision using ens
    bin_lstm_avg=np.zeros((avg_mse_lstm.shape[0],1))# storing binary change decision using lstm
    for i in np.arange(0,avg_mse_lstm.shape[0]):
        if avg_mse_ann[i]>ann_th_avg:
            bin_ann_avg[i,0]=1
        if avg_mse_cnn[i]>cnn_th_avg:
            bin_cnn_avg[i,0]=1
        if avg_mse_ens[i]>ens_th_avg:
            bin_ens_avg[i,0]=1
        if avg_mse_lstm[i]>lstm_th_avg:
            bin_lstm_avg[i,0]=1
    

    
    #----------------change summary for each method separately (if change point, degree/ magnitude, direction (obs-pred))------
    ch_summary=ch_summary_methods(bin_ann_avg,bin_cnn_avg, bin_lstm_avg,bin_ens_avg,avg_pred_ann,avg_pred_cnn,avg_pred_lstm,avg_pred_ens,start_idx, norm_ts_mm, win_l)
    
    ch_steps=np.zeros((observation_count,11))#ntl,ntl-filtered,non_weighted_avg, wt_ntl_filtered, gap-filled, gap-filled-filtered,non_weighted_gap_ wt-gap-filled, flags
    #ts_stack[:,0]=ntl[]
    #ts_stack[:,1]=filterd_ntl
    ch_steps[:,0]=non_nan_yr[start_idx+win_l:]
    ch_steps[:,1]=non_nan_month[start_idx+win_l:]
    ch_steps[:,2]=non_nan_day[start_idx+win_l:]
    ch_steps[:,3]=ch_summary[0,:,0]# bin ch ann
    ch_steps[:,4]=ch_summary[0,:,2]# dir ch
    ch_steps[:,5]=ch_summary[1,:,0]# bin ch cnn
    ch_steps[:,6]=ch_summary[1,:,2]
    ch_steps[:,7]=ch_summary[3,:,0]# bin ch lstm
    ch_steps[:,8]=ch_summary[3,:,2]
    ch_steps[:,9]=ch_summary[2,:,0]# bin ch ens
    ch_steps[:,10]=ch_summary[2,:,2]
    
    np.savetxt(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ch_pt.csv")),ch_steps,fmt='%f', delimiter=',', newline='\n',header='yr, month, day,ch_pt_ann, ch_dir_ann,ch_pt_cnn, ch_dir_cnn, ch_pt_lstm, ch_dir_lstm, ch_pt_ens, ch_dir_ens')
    
    
    #----------------inverse prediction to ntl scale------
    inversed_obs = mm_obj.inverse_transform(norm_ts_mm[win_l+start_idx:])
    inversed_pred_ens = mm_obj.inverse_transform(avg_pred_ens)#using ens
    inversed_pred_ann = mm_obj.inverse_transform(avg_pred_ann)
    inversed_pred_cnn = mm_obj.inverse_transform(avg_pred_cnn)
    inversed_pred_lstm = mm_obj.inverse_transform(avg_pred_lstm)
    
    inv_pred=np.zeros((observation_count,8))#ntl,ntl-filtered,non_weighted_avg, wt_ntl_filtered, gap-filled, gap-filled-filtered,non_weighted_gap_ wt-gap-filled, flags
    #ts_stack[:,0]=ntl[]
    #ts_stack[:,1]=filterd_ntl
    inv_pred[:,0]=non_nan_yr[start_idx+win_l:]
    inv_pred[:,1]=non_nan_month[start_idx+win_l:]
    inv_pred[:,2]=non_nan_day[start_idx+win_l:]
    inv_pred[:,3]=inversed_obs[:,0]
    inv_pred[:,4]=inversed_pred_ens[:,0]
    inv_pred[:,5]=inversed_pred_ann[:,0]
    inv_pred[:,6]=inversed_pred_cnn[:,0]
    inv_pred[:,7]=inversed_pred_lstm[:,0]
    
    #ts_stack[:,6]=np.datetime64(poly_zarr["Dates"][0:])
    
    np.savetxt(str(Path("/app/temp_data",f"fua_{UA}_{tile}_pred_ntl_scale.csv")),inv_pred,fmt='%f', delimiter=',', newline='\n',header='yr, month, day,ntl_avg,pred_ens, pred_ann, pred_cnn, pred_lstm')
    
    
    plt.figure(figsize=(21,11))
    plt.plot(inversed_obs,'k',label='observed')
    plt.plot(inversed_pred_ens,'r',label='predicted')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)
    plt.ylabel('NTL(nW cm$^-$$^2$ sr$^-$$^1$)', fontsize=17)
    #plt.savefig(os.path.join(write_dir_plots,UA+'_ens_pred_with_relu.png'), dpi=180)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ens_pred.png")), dpi=180)
    plt.close()
    
    x=np.arange(0,inversed_obs.shape[0])
    print('lengts',x,inversed_obs.shape[0])
    plt.figure(figsize=(21,11))
    plt.subplot(2,1,1)
    plt.plot(inversed_obs,'k',label='observed')
    plt.plot(inversed_pred_ens,'r',label='predicted')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)
    plt.ylabel('NTL(nW cm$^-$$^2$ sr$^-$$^1$)', fontsize=17)
    plt.subplot(2,1,2)
    plt.scatter(x,c_flag)
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=17)
    #plt.savefig(os.path.join(write_dir_plots,UA+'_ens_pred_with_flag.png'), dpi=180)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ens_pred_flag.png")), dpi=180)
    plt.close()
    #plt.close()
    
    #np.savetxt(os.path.join(write_dir_plots,UA+'_ens_pred.csv'),inversed_pred_ens,fmt='%10.4f', delimiter=',', newline='\n',header='ntl_pred')
    #np.savetxt(os.path.join(write_dir_plots,UA+'_ens_obs.csv'),inversed_obs,fmt='%10.4f', delimiter=',', newline='\n',header='ntl_obs')
    plot_change(ch_summary,avg_pred_ann,avg_pred_cnn,avg_pred_lstm,avg_pred_ens, norm_ts_mm, start_idx,c_flag,win_l, UA,tile, years_tick, years_tick_pos)
    d=eval_dates
    
    rec_0, prec_0, del_0=evaluate_fn_fp_delay(d, UA,tile, num_val, ch_summary,start_idx, end_idx,d_full,win_l,0)
    rec_1, prec_1, del_1=evaluate_fn_fp_delay(d, UA,tile, num_val, ch_summary,start_idx, end_idx,d_full,win_l,1)
    rec_2, prec_2, del_2=evaluate_fn_fp_delay(d, UA,tile, num_val, ch_summary,start_idx, end_idx,d_full,win_l,2)
    rec_3, prec_3, del_3=evaluate_fn_fp_delay(d, UA,tile, num_val, ch_summary,start_idx, end_idx,d_full,win_l,3)
    
    eval_stack=np.zeros((1,12))
    
    eval_stack[0,0]=rec_0
    eval_stack[0,1]=prec_0
    eval_stack[0,2]=del_0
    
    eval_stack[0,3]=rec_1
    eval_stack[0,4]=prec_1
    eval_stack[0,5]=del_1
    
    eval_stack[0,6]=rec_2
    eval_stack[0,7]=prec_2
    eval_stack[0,8]=del_2
    
    eval_stack[0,9]=rec_3
    eval_stack[0,10]=prec_3
    eval_stack[0,11]=del_3
    
    np.savetxt(str(Path("/app/temp_data",f"fua_{UA}_{tile}_eval.csv")),eval_stack,fmt='%f', delimiter=',', newline='\n',header='rec-ann, prec-ann,del-ann,rec-cnn, prec-cnn,del-cnn, rec-ens, prec-ens,del-ens, rec-lstm, prec-lstm,del-lstm ')
    
    print('TOOK:', time()-s)
    