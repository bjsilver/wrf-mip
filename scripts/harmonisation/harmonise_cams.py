#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import xesmf as xe
import xarray as xr
import salem
from constants import common_grid
from glob import glob
import calendar
from tqdm import tqdm
import os

model_path = '/nfs/a340/eebjs/wrf-mip/model_data/cams/'

file_list = glob(model_path+'wrfout_w4chinameic_*_2017.nc')

name_map = {'PM2_5_DRY':'pm25',
            'no2':'no2',
            'o3':'o3',
            'so2':'so2'}

#%% load in data 
def preprocess(ds):
    return ds.loc[{'bottom_top':0}][list(name_map.keys())]

# get example ds for regridder
model_ds = salem.open_mf_wrf_dataset(model_path+'wrfout_d01_2017-06-01*',
                                     preprocess=preprocess)

#%% create regridder

regridder = xe.Regridder(ds_in=model_ds, ds_out=common_grid, method='bilinear')
regridder.to_netcdf('./regridders/wrf-chem.nc')

#%%

def regrid_month(month):

    monthname = calendar.month_abbr[month]
    spath = save_path+f'{monthname}_regridded.nc'
    if os.path.exists(spath):
        print(f'{monthname} already done')
        return
    
    
    model_dss = []
    for day in tqdm(range(1, calendar.monthrange(2017, month)[1]+1)):
        model_ds = salem.open_mf_wrf_dataset(model_path+f'wrfout_d01_2017-{str(month).zfill(2)}-{str(day).zfill(2)}*',
                                             preprocess=preprocess)
        model_dss.append(model_ds)
        
    model_ds = xr.concat(model_dss, dim='time')
    ds = regridder(model_ds)
    
    ds = ds.rename(name_map)
    
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(spath, encoding=encoding)
    print(monthname+' done')

for month in [6,7,8]:
    regrid_month(month)