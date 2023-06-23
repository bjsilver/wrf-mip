#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import geopandas as gpd

common_grid = xr.Dataset(coords={'lon':np.arange(67, 140, .1), 'lat':np.arange(16,56,.1)})

#%% region mask of china

def create_china_regions():
    shp_path = '/nfs/a340/eebjs/shapefiles/'
    chn = gpd.read_file(shp_path+'gadm41_CHN_1.shp')
    chn = chn.set_index('NAME_1')
    
    regionmapper =\
    {'Western China':['Xinjiang Uygur', 'Qinghai', 'Xizang', 'Gansu', 'Ningxia Hui', 'Nei Mongol'],
     'Southwest China':['Sichuan', 'Yunnan', 'Chongqing', 'Guizhou',
                        'Guangxi', 'Hainan'],
     'Southern East China':['Guangdong', 'Hunan', 'Jiangxi', 'Fujian',
                            'Zhejiang', 'Anhui', 'Hubei', 'Shanghai'],
     'Northern East China':['Jiangsu', 'Henan', 'Shaanxi', 'Shanxi',
                            'Hebei', 'Shandong', 'Jiangsu', 'Beijing',
                            'Tianjin', 'Hebei'],
     'Northeast China':['Liaoning', 'Jilin', 'Heilongjiang']}
    
    regions = []
    for region, provinces in regionmapper.items():
        
        merged = chn.loc[provinces].dissolve()
        merged['name'] = region
        merged = merged[['name', 'geometry']]
        regions.append(merged)
        
    regions = pd.concat(regions)
    regions.index=list(range(0, len(regionmapper)))
    return regionmask.from_geopandas(regions,names='name')

china_regions = create_china_regions()
