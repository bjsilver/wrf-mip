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
import pandas as pd

model_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfcmaq/raw/'
save_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfcmaq/regridded/'

file_list = glob(model_path+'CCTM_D51a_2017*_COMBINE_ACONC.CMAQ51-BENCHMARK')

name_map = {'PM25_TOT':'pm25',
            'NO2':'no2',
            'O3':'o3',
            'SO2':'so2'}

# load in grid file
grid = xr.open_dataset(model_path+'GRIDCRO2D_20170601')

#%% load in data 
def preprocess(ds):
    ds = ds[list(name_map.keys())].loc[{'LAY':0}]
    return ds

# def load_wrfcmaq_data():
# get example ds for regridder
model_ds = xr.open_mfdataset(file_list, concat_dim='TSTEP', preprocess=preprocess, combine='nested')

timestamps = pd.date_range('2017-06-01', '2017-08-31 23:00', freq='H')

model_ds = model_ds.assign_coords({'TSTEP':timestamps})
model_ds = model_ds.rename({'ROW':'y', 'COL':'x', 'TSTEP':'time',})

model_ds = model_ds.assign_coords({
    "lat": (["y", "x"], grid['LAT'][0,0].values),
    "lon": (["y", "x"], grid['LON'][0,0].values)
})

## NOT WORKING SORT!!!!


#%% create regridder

regridder = xe.Regridder(ds_in=model_ds, ds_out=common_grid, method='bilinear')
regridder.to_netcdf('./regridders/wrfcmaq.nc')

#%% regrid

regridded = regridder(model_ds)
regridded = regridded.rename(name_map)
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in regridded.data_vars}
regridded.to_netcdf(save_path+'wrfcmaq_regridded.nc',
                    encoding=encoding)
