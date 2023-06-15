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

model_path = '/nfs/a336/libclsr/AIA/wrfchem_output/w4chinameic2017/'
save_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/'

file_list = glob(model_path+'wrfout_w4chinameic_*_2017.nc')

name_map = {'pm25':'PM2_5_DRY',
            'no2':'no2',
            'o3':'o3',
            'so2':'so2'}

#%% load in data 
def preprocess(ds):
    return ds.loc[{'bottom_top':0}][list(name_map.values())]

# get example ds for regridder
model_ds = salem.open_mf_wrf_dataset(model_path+'wrfout_d01_2017-06-01*',
                                     preprocess=preprocess)

#%% create regridder

regridder = xe.Regridder(ds_in=model_ds, ds_out=common_grid, method='bilinear')
regridder.to_netcdf('./regridders/wrf-chem.nc')

#%%

# def regrid_month(month):
month = 5
monthname = calendar.month_abbr[5]

model_ds = salem.open_mf_wrf_dataset(model_path+f'wrfout_d01_2017-{str(month).zfill(2)}-*',
                                     preprocess=preprocess)

ds = regridder(model_ds)
