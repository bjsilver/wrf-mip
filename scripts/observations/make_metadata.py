#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:27:58 2023

@author: eebjs
"""

import scipy.io
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import ruptures as rpt
import os

cnemc_path = '/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/'

def get_latlons():
    mat = scipy.io.loadmat(cnemc_path+'latlon.mat')['latlon']
    df = pd.DataFrame(mat).T
    df = df.rename({0:'lon', 1:'lat', 2:'id'}, axis=1)
    df = df.set_index('id')
    df.index = df.index.astype(int)
    return df


meta = get_latlons()

for pol in ['no2', 'ozone', 'pm2_5', 'so2']:

    cleaned = pd.read_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/cleaned/{pol}_obs.csv', index_col=0)
    flags =  pd.read_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/flags/{pol}_flags.csv', index_col=0)
#%%
    for station in meta.index:
        if str(station) in cleaned.columns:
            meta.loc[station, f'{pol}'] = True
        else:
            meta.loc[station, f'{pol}'] = False