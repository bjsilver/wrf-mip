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
save_path = '/nfs/a340/eebjs/wrf-mip/model_data/cams/regridded'

name_map = {'pm2p5':'pm25',
            'no2':'no2',
            'go3':'o3',
            'so2':'so2'}

#%% load in data 


# get example ds for regridder
surface = xr.open_dataset(model_path+'CAMS_reanalysis.grib', engine="cfgrib",
                           filter_by_keys={'typeOfLevel': 'surface'})
hybrid = xr.open_dataset(model_path+'CAMS_reanalysis.grib', engine="cfgrib",
                           filter_by_keys={'typeOfLevel': 'hybrid'})

gribds = xr.merge([surface,hybrid])
gribds = gribds.drop(['number', 'step','surface','valid_time','hybrid'])

#%% create regridder

regridder = xe.Regridder(ds_in=gribds, ds_out=common_grid, method='bilinear')
regridder.to_netcdf('./regridders/cams.nc')

#%%

regridded = regridder(gribds)
regridded = regridded.rename(name_map)

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in regridded.data_vars}
regridded.to_netcdf('/nfs/a340/eebjs/wrf-mip/model_data/cams/regridded/cams_regridded.nc', encoding=encoding)