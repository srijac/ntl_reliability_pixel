import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.seasonal import STL
#from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.api as sm
#from sklearn.metrics import mean_squared_error
import os
import math
from pathlib import Path, PureWindowsPath
    
def plot_change(ch_summary,avg_pred_ann,avg_pred_cnn,avg_pred_lstm,avg_pred_ens, norm_ts_mm, start_idx,conf, win_l, UA,tile,years_tick, years_tick_pos):
    plt.figure(figsize=(21,11))
    plt.subplot(2,1,1)
    plt.plot(ch_summary[0,:,2],'b', label='ann')
    plt.plot(ch_summary[1,:,2],'k', label='cnn')
    plt.plot(ch_summary[2,:,2],'r', label='ens')
    plt.plot(ch_summary[3,:,2],'g',label='lstm')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=12)
    plt.ylabel('direction,magnitude(normalized)')
    plt.legend()
    #plt.axhline(y=top_k_val_ann_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-a')
    #plt.axhline(y=top_k_val_cnn_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-c')
    #plt.axhline(y=top_k_val_ens_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-e')
    plt.subplot(2,1,2)
    plt.plot(avg_pred_ann,'b',label='ann')#marker='*'
    plt.plot(avg_pred_cnn,'k',label='cnn')#marker='o'
    plt.plot(avg_pred_ens,'r',label='ens')
    plt.plot(avg_pred_lstm,'g',label='lstm')
    plt.plot(norm_ts_mm[win_l+start_idx:])
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=17)
    plt.legend()
    plt.ylabel('Predictions')
    #plt.savefig(os.path.join(write_dir_plots,UA+'_ch_direction_with_relu.png'), dpi=180)
    #str(Path("/app/temp_data",f"fua_{UA}_{tile}_recovery_segments.csv"))
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ch_direction_pred.png")), dpi=180)
    #plt.savefig('gl_city_subset/change_dir/'+UA+'-ch_dir_trial1_all_2019_norelu_v2.png', dpi = 180) 
    plt.close()
    
    '''plt.figure(figsize=(21,11))
    plt.subplot(5,1,1)
    plt.plot(ch_summary[2,:,2],'r', label='ens')
    plt.plot(ch_summary[0,:,2],'b', label='ann')
    plt.plot(ch_summary[1,:,2],'k', label='cnn')
    plt.plot(ch_summary[3,:,2],'g', label='lstm')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=12)
    plt.ylabel('direction, magnitude(normalized)')
    plt.legend()
    #plt.axhline(y=top_k_val_ann_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-a')
    #plt.axhline(y=top_k_val_cnn_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-c')
    #plt.axhline(y=top_k_val_ens_avg[6], linestyle='-.',label=str(top_k_percent[idx])+'-e')
    plt.subplot(5,1,2)
    #plt.plot(avg_pred_ann,'b',label='ann')#marker='*'
    #plt.plot(avg_pred_cnn,'k',label='cnn')#marker='o'
    plt.plot(avg_pred_ens,'r',label='ens')
    #plt.plot(avg_pred_lstm,'g',label='lstm')
    plt.plot(norm_ts_mm[win_l+start_idx:],'m',label='data')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.ylabel('Prediction Ens')
    plt.legend()
    plt.subplot(5,1,3)
    plt.plot(avg_pred_ann,'b',label='ann')#marker='*'
    plt.plot(avg_pred_cnn,'k',label='cnn')#marker='o'
    plt.plot(avg_pred_ens,'r',label='ens')
    plt.plot(avg_pred_lstm,'g',label='lstm')
    plt.plot(norm_ts_mm[win_l+start_idx:],'m',label='data')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.ylabel('Prediction Ens')
    plt.legend()
    plt.subplot(5,1,4)
    plt.plot(conf)
    plt.ylabel('Ens Confidence')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.subplot(5,1,5)
    plt.plot(perc_ch_ntl)
    plt.ylabel('Ens Percentage change (ntl scale)')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    
    #plt.savefig(os.path.join(write_dir_plots,UA+'_ch_direction_conf_perc_ch_with_relu.png'), dpi=180)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ch_dir_pred_conf.png")), dpi=180)
    #plt.savefig('gl_city_subset/change_dir_ENS/'+UA+'-ch_dir_trial1_all_2019_norelu_v2.png', dpi = 180) 
    #plt.show()
    plt.close() '''
    

def pred_plots(avg_pred_ann, avg_pred_cnn, avg_pred_ens, avg_pred_lstm,norm_ts_mm, win_l,start_idx,years_tick, years_tick_pos, UA, tile, avg_mse_ens):
    plt.figure(figsize=(12,5))
    plt.subplot(2,1,1)
    plt.plot(avg_pred_ann,color='b',label='ann')
    plt.plot(avg_pred_cnn,color='k',label='cnn')
    plt.plot(avg_pred_ens,color='r',label='ens')
    plt.plot(avg_pred_lstm,color='g',label='lstm')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=12)
    plt.legend()
    #plt.plot(pred_m_ann[:,0])
    plt.plot(norm_ts_mm[win_l+start_idx:], color='m')
    plt.subplot(2,1,2)
    #plt.plot(avg_mse_ann,color='b',label='ann')
    #plt.plot(avg_mse_cnn,color='k',label='cnn')
    plt.plot(avg_mse_ens,color='r',label='ens')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=12)
    #plt.plot(avg_mse_lstm,color='g',label='lstm')
    plt.legend()
    #plt.savefig(os.path.join(write_dir_plots,UA+'_pred_ens_err_with_relu.png'), dpi=180)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_pred_ens_err.png")), dpi=180)
    #plt.savefig('gl_city_subset/change_pred/'+UA+'-trial1_all_compare_norelu_v2.png', dpi = 180) 
    #plt.show()
    plt.close()
    
    
def plot_threshold(avg_mse_ann, avg_mse_cnn, avg_mse_ens, avg_mse_lstm, years_tick, years_tick_pos, UA, tile, top_k_val_ann_avg, top_k_val_cnn_avg, top_k_val_ens_avg,top_k_val_lstm_avg,cols,top_k_percent):
    plt.figure(figsize=(21,11))
    plt.subplot(4,1,1)
    plt.plot(avg_mse_ann,color='blueviolet',label='ann')
    for idx, val in enumerate(top_k_val_ann_avg):
        plt.axhline(y=val, linestyle='-.',color=cols[idx],label=str(top_k_percent[idx])+'-a')
        plt.legend(ncol=len(top_k_val_ann_avg)+1)
    #plt.title('multi ann avg mse')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.subplot(4,1,2)
    plt.plot(avg_mse_cnn,color='blueviolet',label='cnn')
    for idx, val in enumerate(top_k_val_cnn_avg):
        plt.axhline(y=val, linestyle='-.',color=cols[idx],label=str(top_k_percent[idx])+'-c')
        plt.legend(ncol=len(top_k_val_cnn_avg)+1)
    #plt.title('multi cnn avg mse')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.subplot(4,1,3)
    plt.plot(avg_mse_ens,color='blueviolet',label='ens')
    for idx, val in enumerate(top_k_val_ens_avg):
        plt.axhline(y=val, linestyle='-.',color=cols[idx],label=str(top_k_percent[idx])+'-e')
        plt.legend(ncol=len(top_k_val_ens_avg)+1)
    #plt.title('multi ens avg mse')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    plt.subplot(4,1,4)
    plt.plot(avg_mse_lstm,color='blueviolet',label='lstm')
    for idx, val in enumerate(top_k_val_lstm_avg):
        plt.axhline(y=val, linestyle='-.',color=cols[idx],label=str(top_k_percent[idx])+'-e')
        plt.legend(ncol=len(top_k_val_lstm_avg)+1)
    #plt.title('multi lstm avg mse')
    plt.xticks(ticks = years_tick_pos ,labels = years_tick, rotation = 0, fontsize=15)
    #plt.savefig(os.path.join(write_dir_plots,UA+'_pred_err_with_relu.png'), dpi=180)
    plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_pred_err_th_v3.png")), dpi=180)
    #plt.savefig('gl_city_subset/change_pred/'+UA+'-pred_err_trial1_all_2019_norelu_v2.png', dpi = 180) 
    plt.close()
    
