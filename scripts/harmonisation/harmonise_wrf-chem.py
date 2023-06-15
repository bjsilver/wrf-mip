#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import xesmf as xe

import xarray as xr
import xwrf
from constants import common_grid
from glob import glob

model_path = '/nfs/a336/libclsr/AIA/wrfchem_output/w4chinameic2017/'


file_list = glob(model_path+'wrfout_w4chinameic_*_2017.nc')

name_map = {'pm25':'PM2_5_DRY',
            'no2':'no2',
            'o3':'o3',
            'so2':'so2'}

def preprocess(ds):
    variables = list(name_map.values())
    return ds[variables]

ds = xr.open_mfdataset(
    file_list,
    parallel=True,
    concat_dim="Time",
    combine="nested",
    preprocess=preprocess,
    chunks={'Time': 1},
)

xwrf.tutorial.open_dataset(file_list[0])
