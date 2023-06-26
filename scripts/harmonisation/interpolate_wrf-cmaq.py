#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from constants import pollutants


cmaq_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfcmaq/regridded/'
cmaq = xr.open_dataset('/nfs/a340/eebjs/wrf-mip/model_data/wrfcmaq/regridded/wrfcmaq_regridded.nc')

measurement_path = '/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/'
meta = pd.read_csv(measurement_path+'metadata.csv', index_col=0)

spath = measurement_path + 'wrfcmaq_interped/'


def interpolate_wrfcmaq(cmaq, pol):

    da = cmaq[pol]
    
    tindex = da.time.to_pandas().index.tz_localize('UTC')
    timemap = pd.Series(tindex)
    
    points = (timemap.index.values, da.lat.values, da.lon.values)
    interper = RegularGridInterpolator(points=points, values=da.values)

    srs = []
    for idn in tqdm(meta.index):
        if idn == 1590:
            continue
        lat, lon = meta.loc[idn, ['lat', 'lon']]
        if pd.isnull(lat):
            continue
        interped = interper((timemap.index.values, lat, lon))
        sr =  pd.Series(interped, index=tindex)
        sr.name = idn
        srs.append(sr)
        
    df = pd.concat(srs, axis=1)
        
    df.to_csv(spath+f'{pol}_interped.csv')
    
for pol in pollutants:
    interpolate_wrfcmaq(cmaq, pol=pol)