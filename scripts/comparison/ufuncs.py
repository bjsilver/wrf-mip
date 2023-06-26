#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:46:33 2023

@author: eebjs
"""

import xarray as xr


def load_models():
    
    #wrfchem
    wrf_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/regridded/'
    wrfchem = xr.open_dataset(wrf_path+'wrfchem_regridded.nc')
    
    #wrf cmaq
    wrf_cmaq = xr.open_dataset('/nfs/a340/eebjs/wrf-mip/model_data/wrfcmaq/regridded/wrfcmaq_regridded.nc')
    
    
    d = {
        'WRF-Chem':wrfchem,
        'WRF-CMAQ':wrf_cmaq
        }
    
    return d