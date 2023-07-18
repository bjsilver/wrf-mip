#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:07:11 2023

@author: ben
"""

from constants import china_regions
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import geopandas as gpd

measurement_path = '/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/'
meta = pd.read_csv(measurement_path+'metadata.csv', index_col=0)
meta.index = meta.index.astype(str)
# drop dodgy
meta = meta.drop('1590')


gdf = china_regions.to_geodataframe()
points = gpd.GeoSeries(index=meta.index)
for idn in tqdm(meta.index):
    
    lat, lon = meta.loc[idn, ['lat', 'lon']]
    point = Point(lon, lat)
    points.loc[idn] = point


for region in china_regions.names:

    geom = gdf.loc[gdf.names==region].geometry.values[0]
    clipped = gpd.clip(points, geom)
    meta.loc[clipped.index, 'region'] = region
    
meta.to_csv(measurement_path+'metadata_with_region.csv')
