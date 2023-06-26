#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import numpy as np
import xarray as xr

common_grid = xr.Dataset(coords={'lon':np.arange(67, 140, .1), 'lat':np.arange(16,56,.1)})

pollutants = ['pm25', 'o3', 'so2', 'no2']
