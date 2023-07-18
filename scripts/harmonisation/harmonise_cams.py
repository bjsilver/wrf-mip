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


#%% conversions:
    
# convert pm2p5 kg/m3 to ug/m3
gribds['pm2p5'] = gribds['pm2p5'] * 1e9

# convert mass mixing ratios (kg/kg) to concentration (ug/m3)
def mmr_to_conc(da, p, T):

    R = 8.314
    M_air = 0.02897

    conv = da * (   (p * M_air)   / (R   *  T)     ) * 1e9
    return conv

for var in ['no2', 'go3', 'so2']:
    gribds[var] = mmr_to_conc(gribds[var], gribds['sp'], gribds['t'])


    
#%% create regridder

regridder = xe.Regridder(ds_in=gribds, ds_out=common_grid, method='bilinear')
regridder.to_netcdf('./regridders/cams.nc')

#%%

regridded = regridder(gribds[name_map.keys()])
regridded = regridded.rename(name_map)

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in regridded.data_vars}
regridded.to_netcdf('/nfs/a340/eebjs/wrf-mip/model_data/cams/regridded/cams_regridded.nc', encoding=encoding)