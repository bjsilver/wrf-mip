#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import numpy as np
import xarray as xr
import pandas as pd
import regionmask

common_grid = xr.Dataset(coords={'lon':np.arange(67, 140, .1), 'lat':np.arange(16,56,.1)})

pollutants = ['pm25', 'o3', 'so2', 'no2']

models = ['wrfchem', 'wrfcmaq']

pretty_model = {'wrfchem':'WRF-Chem',
                'wrfcmaq':'WRF-CMAQ'}

pretty_pol = {'pm25':'$\mathrm{PM_{2.5}}$',
              'no2':'$\mathrm{NO_2}$',
              'so2':'$\mathrm{SO_2}$',
              'o3':'$\mathrm{O_3}$'}

#%% region mask of china

def create_china_regions():
    shp_path = '/nfs/a340/eebjs/shapefiles/'
    chn = gpd.read_file(shp_path+'gadm41_CHN_1.shp')
    chn = chn.set_index('NAME_1')
    
    regionmapper =\
    {'Western China':['Xinjiang Uygur', 'Qinghai', 'Xizang', 'Gansu', 'Ningxia Hui', 'Nei Mongol'],
     ###########################################################
     'Northern East China':['Jiangsu', 'Henan', 'Shaanxi', 'Shanxi', 'Anhui',
                            'Hebei', 'Shandong', 'Jiangsu', 'Beijing',
                            'Tianjin', 'Hebei'],
     ##############################################################
     'Northeast China':['Liaoning', 'Jilin', 'Heilongjiang'],
     ###########################################################
     'Southwest China':['Sichuan', 'Yunnan', 'Chongqing', 'Guizhou',
                        'Guangxi', 'Hainan'],
     ##############################################################
     'Southern East China':['Guangdong', 'Hunan', 'Jiangxi', 'Fujian',
                            'Zhejiang', 'Hubei', 'Shanghai']}
    
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