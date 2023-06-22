#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import xarray as xr

wrf_path = '/nfs/a340/eebjs/wrf-mip/model_data/wrfchem/regridded/'
wrf = xr.open_mfdataset(wrf_path+'*_regridded.nc')

measurement_path = '/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/'