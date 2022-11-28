from pathlib import Path
import zarr
import numpy as np
from datetime import datetime, timedelta
import rclone
import os 
import math
import glob 
import warnings
#import zarr
import os
import pickle
from itertools import islice
import json
from shutil import rmtree
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from ts_main_v3 import *

#tile_zarr, pix_r,pix_c, gf_ntl, global_r, global_c

#def specify_datasets(zarr_obj, elec_row, elec_col, global_pixel_vs, global_pixel_hs, coords):
def specify_datasets(zarr_obj, pix_r,pix_c, gf_ntl, global_r, global_c, zarr_date):
    zarr_obj.create_dataset("Pixel_V",
                            data=pix_r,
                            shape=(1, len(pix_r)),
                            dtype="uint16")
    zarr_obj.create_dataset("Pixel_H",
                            data=pix_c,
                            shape=(1, len(pix_c)),
                            dtype="uint16")
    zarr_obj.create_dataset("Global_Pixel_V",
                            data=global_r,
                            shape=(1, len(global_r)),
                            dtype="uint16")
    zarr_obj.create_dataset("Global_Pixel_H",
                            data=global_c,
                            shape=(1, len(global_c)),
                            dtype="uint16")
    zarr_obj.create_dataset("Gap_Filled_DNB_BRDF-Corrected_NTL",
                            data=gf_ntl,
                            shape=(gf_ntl.shape[0], gf_ntl.shape[1]),
                            dtype="float")
    zarr_obj.create_dataset("Dates",
                            data=zarr_date,
                            shape=(1, len(zarr_date)),
                            dtype="datetime64[D]")

def create_zarr(storage_path):
    store = zarr.DirectoryStore(storage_path)
    root = zarr.group(store=store, overwrite=True)
    return root
    
def create_s3_dir(config, uri):

    rclone.with_config(config).run_cmd(command="mkdir", extra_args=[uri, "--quiet"])

def main():
    
    # SETUP
    
    # Start the clock
    #stime = time()     
    #print(f"Start time:{stime}.")
    # List for tiles to be downloaded
    '''tile_list = []
    
    # Import the list of tiles
    with open(Path(f"/app/{tile_file}.txt"), 'r') as f:
        for line in f:
            tile_list.append(line.strip('\n'))
    
    # Update
    print(f"Starting zarr conversion of {len(tile_list)} tiles. Configuring s3.")'''
    
    # Get s3 secrets
    with open(Path("/app/s3/s3accesskey2.txt"), 'r') as f:
        for line in f:
            s3_access_key = str(line.strip()[4:-1])
            break
            
    with open(Path("/app/s3/s3secretkey2.txt"), 'r') as f:
        for line in f:
            s3_secret_key = str(line.strip()[4:-1])
            break
            
    with open(Path("/app/tile_patch.json"), 'r') as f:
        tile_patch = json.load(f)

        
    #train_block=tile_patch[root_tile][0]['train']
    #test_block=tile_patch[root_tile][0]['test']
    
    #print('train, test', train_block, test_block)
    
    '''bounds=get_tile_patch(root_tile,train_block)
    if len(bounds)==3:
        rw=bounds[2]
        cw=bounds[2]
    elif len(bounds)==4:
        rw=bounds[2]
        cw=bounds[3]
    print('bounds:',bounds[0],bounds[1],bounds[2], rw, cw)'''
    
    city='Houston'
    t='h08v06'
    
    u_row=tile_patch[t][0][city][0]
    u_col=tile_patch[t][0][city][1]
    l_row=tile_patch[t][0][city][2]
    l_col=tile_patch[t][0][city][3]
    
    end_date=tile_patch[t][0]['end_change_date']
    
    print('r c bounds:', u_row, u_col, l_row, l_col)
    u_row_ad=math.floor(u_row/2)
    u_col_ad=math.floor(u_col/2)
    l_row_ad=math.floor(l_row/2)
    l_col_ad=math.floor(l_col/2)
    print('r c bounds:', u_row_ad, u_col_ad, l_row_ad, l_col_ad)
           
    #with open(Path("/app/Training_dates.csv"), 'r') as f:
    #training_len=pd.read_csv(Path("/app/training_dates.csv"), 'r')
    
    # Form a remote configuration for rclone
    cfg = """[ceph]
    type = s3
    provider = Ceph Object Storage
    endpoint = http://rook-ceph-rgw-nautiluss3.rook
    access_key_id = {0}
    secret_access_key = {1}
    region =
    nounc = true"""
    
    # Add the s3 secrets to the configuration
    cfg = cfg.format(s3_access_key, s3_secret_key)
    
    for new_dir in ["ceph:zarrs/wsf/wsf_1km/pixel_rel"]:#
        create_s3_dir(cfg, new_dir)
    
    # Make s3 "directories" for the output data
    '''for new_dir in ["ceph:fua_subset_numpy"]:#Fua_run2 is the main dir? with fua as subdir?
        create_s3_dir(cfg, new_dir)'''
    '''w_dir="ceph:anomaly_write_global"
    w_dir_wt="ceph:anomaly_write_global/weights"
    w_dir_fc="ceph:anomaly_write_global/forecast"
    w_dir_comp="ceph:anomaly_write_global/composite"
    w_dir_plots="ceph:anomaly_write_global/plots"
    for new_dir in [w_dir, w_dir_wt, w_dir_fc,w_dir_comp, w_dir_plots ]:
        create_s3_dir(cfg, new_dir)'''
        
    #tile_list=['h07v06','h09v05','h08v06']
    tile_list=['h08v06']
    
    sample_pix_h=1035
    sample_pix_v=29
    sample_pix_h_ad=math.floor(sample_pix_h/2)
    sample_pix_v_ad=math.floor(sample_pix_v/2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for tile in tile_list:
            '''existing_ds_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:zarrs/wsf/wsf_1km/{tile}"])
            rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"ceph:zarrs/wsf/wsf_1km/{tile}", str(Path(f"/app/temp_data/"))])'''
            #rclone.with_config(cfg).run_cmd(command='copy', extra_args=[f'ceph:zarrs/wsf/wsf_1km/wsf_{tile}.zarr', f'/wsf_1km/wsf_{tile}.zarr'])
            rclone.with_config(cfg).run_cmd(command='copy', extra_args=[f'ceph:zarrs/wsf/wsf_1km/wsf_{tile}.zarr', f'/app/temp_data/wsf_{tile}.zarr'])
        
        
            # Specify a path to a zarr file (root of the directory-like structure)
            #zarr_path = f"/wsf_1km/wsf_{tile}.zarr"
            zarr_path = f"/app/temp_data/wsf_{tile}.zarr"

            # Open the zarr file in read mode
            poly_zarr = zarr.open(zarr_path, mode='r')

            # Get the tile and polygon name (just stripping down the path, nothing to learn here!)
            split_path = str(zarr_path).split('\\')
            split_name = split_path[-1].split('_')
            tile_name = split_name[-1].strip('.zarr')

            # Get the Gap-Filled Night Time Lights
            final_gapfilled_ntl = np.array(poly_zarr["Gap_Filled_DNB_BRDF-Corrected_NTL"])
            print('looking at:', tile)
            print(poly_zarr.tree())
            print('time-series length:', final_gapfilled_ntl.shape[0])
            print('-------------------------------')
            sample_h = np.array(poly_zarr['Pixel_H'])
            sample_v = np.array(poly_zarr['Pixel_V'])
            zarr_date=np.array(poly_zarr['Dates'])
            print('date:',zarr_date[0], zarr_date[3], zarr_date[-1])
            #print(zarr_date.dtype, sample_h.dtype)
            gf_ntl=[]
            pix_r=[]
            pix_c=[]
            global_r=[]
            global_c=[]
            idx_list=[]
            date=[]
            
            tile_v = int(tile_name.split('v')[-1])
            tile_h = int(tile_name.split('v')[0][1:])
            
            print('bounds u col, lcol, urow, lrow:', u_col, l_col,u_row, l_row)
            print('bounds u col, lcol, urow, lrow:', u_col_ad, l_col_ad,u_row_ad, l_row_ad)
            
            for idx,coord in enumerate(zip(sample_h,sample_v)):
                #print('coord', coord, coord[0], coord[1], coord[0]+1)
                if (((coord[0]>=u_col_ad) &(coord[0]<=l_col_ad)) & ((coord[1]>=u_row_ad) &(coord[1]<=l_row_ad))):
                    #print('in index:', i, coo )
                    pix_r.append(coord[1])
                    pix_c.append(coord[0])
                    gf_ntl.append(final_gapfilled_ntl[:,idx])
                    global_r.append((tile_v * 1200) + coord[1])
                    global_c.append((tile_h * 1200) + coord[0])
                    
                    
                    if ((coord[0]==sample_pix_h_ad)&(coord[1]==sample_pix_v_ad)):
                        print('-------------sample match at:-------------',idx, coord[0],coord[1])
            
            pix_r=np.transpose(np.asarray(pix_r))
            pix_c=np.transpose(np.asarray(pix_c))
            gf_ntl=np.transpose(np.asarray(gf_ntl))
            global_r=np.transpose(np.asarray(global_r))
            global_c=np.transpose(np.asarray(global_c))
                    
            print('shapes', pix_r.shape,pix_c.shape,gf_ntl.shape,global_r.shape,global_c.shape, len(pix_r), len(pix_c), len(global_r), len(global_c))
            print('r:',np.min(pix_r), np.max(pix_r), np.min(pix_c), np.max(pix_c))
            
            #tile_zarr = create_zarr(f"/wsf_1km/change_{tile}_{city}.zarr")
            tile_zarr = create_zarr(f"/app/temp_data/change_{tile}_{city}.zarr")

            # Instantiate the datasets for the polygon's zarr
            specify_datasets(tile_zarr, pix_r,pix_c, gf_ntl, global_r, global_c, zarr_date)

            '''rclone.with_config(cfg).run_cmd(command="copy", extra_args=[f"/wsf_1km/change_{tile}_{city}.zarr", f"ceph:zarrs/wsf/wsf_1km/pixel_rel/change_{tile}_{city}.zarr"])
            rmtree(str(Path(f"/wsf_1km/wsf_{tile}.zarr")))
            rmtree(str(Path(f"/wsf_1km/change_{tile}_{city}.zarr")))'''
            
            rclone.with_config(cfg).run_cmd(command="copy", extra_args=[f"/app/temp_data/change_{tile}_{city}.zarr", f"ceph:zarrs/wsf/wsf_1km/pixel_rel/change_{tile}_{city}.zarr"])
            rmtree(str(Path(f"/app/temp_data/wsf_{tile}.zarr")))
            rmtree(str(Path(f"/app/temp_data/change_{tile}_{city}.zarr")))
        
    
    print('done')
    #zarrs/wsf/wsf_1km/pixel_rel/change_{tile}_{city}.zarr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        #rclone.with_config(cfg).run_cmd(command='copy', extra_args=[f"ceph:zarrs/wsf/wsf_1km/pixel_rel/change_{tile}_{city}.zarr", f"/wsf_1km/change_{tile}_{city}.zarr"])#/app/temp_data/
        rclone.with_config(cfg).run_cmd(command='copy', extra_args=[f"ceph:zarrs/wsf/wsf_1km/pixel_rel/change_{tile}_{city}.zarr", f"/app/temp_data/change_{tile}_{city}.zarr"])
        zarr_path = f"/app/temp_data/change_{tile}_{city}.zarr"

        # Open the zarr file in read mode
        poly_zarr = zarr.open(zarr_path, mode='r')

        # Get the tile and polygon name (just stripping down the path, nothing to learn here!)
        split_path = str(zarr_path).split('\\')
        split_name = split_path[-1].split('_')
        tile_name = split_name[-1].strip('.zarr')

        # Get the Gap-Filled Night Time Lights
        gf_ntl = np.array(poly_zarr["Gap_Filled_DNB_BRDF-Corrected_NTL"])
        print('looking at:', tile)
        print(poly_zarr.tree())
        print('time-series length:',gf_ntl.shape[0])
        print('-------------------------------')
        sample_h = np.array(poly_zarr['Pixel_H'])
        sample_v = np.array(poly_zarr['Pixel_V'])
        zarr_date = np.array(poly_zarr['Dates'])
        print('coord start end:',np.min(sample_h), np.max(sample_h), np.min(sample_v), np.max(sample_v))
        print('looking for:', sample_pix_h_ad, sample_pix_v_ad)
    
        for idx,coord in enumerate(zip(sample_h,sample_v)):
            #print('coord', coord, coord[0], coord[1], coord[0]+1)
            #print('list:',idx, coord)
            if ((coord[0]==sample_pix_h_ad)&(coord[1]==sample_pix_v_ad)):
                print('found:', idx, coord)
                plt.figure(figsize=(21,11))
                plt.plot(gf_ntl[:,idx])
                #plt.savefig(str(Path("/app/temp_data",f"fua_{UA}_{tile}_ch_dir_pred_conf.png")), dpi=180)/wsf_1km/pixel_rel/change_{tile}_{city}.zarr
                plt.savefig(str(Path(f"/app/temp_data",f"chplot_{tile}_{city}_{sample_pix_v}_{sample_pix_h}.png")), dpi=180)
                plt.close()
    
                rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path(f"/app/temp_data",f"chplot_{tile}_{city}_{sample_pix_v}_{sample_pix_h}.png")),
                                                        f"ceph:zarrs/wsf/wsf_1km/pixel_rel/chplot_{tile}_{city}_{sample_pix_v}_{sample_pix_h}.png"])    
                ts=gf_ntl[:,idx]
                
    end_date_search=np.array(np.datetime64(end_date))
    print('date:',zarr_date[0], zarr_date[3], zarr_date[-1], np.datetime64(zarr_date[0]).astype(object).year,np.datetime64(zarr_date[0]).astype(object).month, np.datetime64(zarr_date[0]).astype(object).day )
    for idx,x in enumerate(zarr_date):
        x_m=np.array(np.datetime64(x))
        if end_date_search==x_m:
            print('end date:', idx, x, zarr_date[idx])
            end_date_idx=idx
            break
      
        
    end_d=np.datetime64(zarr_date[idx]).astype(object).day
    end_y=np.datetime64(zarr_date[idx]).astype(object).year
    end_m=np.datetime64(zarr_date[idx]).astype(object).month
    
    forecast_city_list(sample_pix_v, sample_pix_h, tile, end_d,end_m,end_y, zarr_path_obs, ts, w_dir_wt, w_dir_fc,w_dir_comp)
    print('done plot')
    os.remove(str(Path(f"/app/temp_data", f"chplot_{tile}_{city}_{sample_pix_v}_{sample_pix_h}.png")))
    rmtree(str(Path(f"/app/temp_data/change_{tile}_{city}.zarr")))
            
    '''os.remove(str(Path(f"/wsf_1km", f"chplot_{tile}_{city}.png")))
    rmtree(str(Path(f"/wsf_1km/change_{tile}_{city}.zarr")))'''
    
    
    
'''if __name__ == "__main__":
    
    # Get the system argument for the tile list
    tile_file = argv[1:][0]   
    
    print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    main(tile_file)'''

if __name__ == "__main__":

    main()
    

