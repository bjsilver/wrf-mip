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
import molmass

model_path = '/nfs/a336/libclsr/AIA/wrfchem_output/w4chinameic2017/'
save_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/regridded/'

file_list = glob(model_path+'wrfout_w4chinameic_*_2017.nc')

name_map = {'PM2_5_DRY':'pm25',
            'no2':'no2',
            'o3':'o3',
            'so2':'so2'}

# unit conversion function
def ppm_to_ugm3 (da, da_T2, da_PSFC):

    name = da.name
    new_attrs = da.attrs

    # get variables needed
    chem = da.name.upper()
    M = molmass.Formula(chem).mass
    R = 8.3145

    # convert unit ppmv -> µg/m³
    da = (M * da_PSFC * da) / (R * da_T2)

    # update attributes with new units
    new_attrs['units'] = 'ug m^-3'
    new_attrs['description'] = chem + \
                               ' mixing ratio converted to mass concentration'
    da.attrs = new_attrs
    da.name = name

    return(da)

#%% load in data 
def preprocess(ds):
    
    ds = ds.loc[{'bottom_top':0}]
    
    das = []
    for pol in ['PM2_5_DRY', 'no2', 'o3', 'so2']:

        da = ds[pol]
        
        if da.units == 'ppmv':
            da = ppm_to_ugm3(da, ds['T2'], ds['PSFC'])
            
        das.append(da)
    
    return xr.merge(das)

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
    
    
#%% combine into one file
wrf_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/regridded/'
wrfchem = xr.open_mfdataset(wrf_path+'*_regridded.nc')
wrfchem = wrfchem.persist()

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in wrfchem.data_vars}
wrfchem.to_netcdf('/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/regridded/wrfchem_regridded.nc', encoding=encoding)