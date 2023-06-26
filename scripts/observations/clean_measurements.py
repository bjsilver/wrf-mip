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


def open_cnemc_year(pollutant, year):
    # open the matlab file
    mat = scipy.io.loadmat(cnemc_path+f'{pollutant}{year}.mat')[f'{pollutant}{year}']
    
    # reformat as dataframe
    df = pd.DataFrame(mat)
    
    # create time index
    tindex = pd.date_range(f'{year}-01-01', f'{year+1}-01-01',
                           freq='H', tz='Asia/Shanghai')[:-1]
    df.index = tindex
    
    # set -999 as NaN
    df = df.where(df!=-999, np.nan)

    latlons = get_latlons()
    df.columns = latlons.index

    return df

def open_cnemc_df(pollutant):
    
    dfs = []
    for year in range(2014, 2020):
        df = open_cnemc_year(pollutant, year)
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def remove_extremes(sr, zscore_threshold=6):
    
    zscores = stats.zscore(sr, nan_policy='omit')#
    return sr.where(zscores < zscore_threshold)
    

# sr = df[1274]
# sr_flags = flags[1274]
def plot_flags(sr, sr_flags):
    
    # plot areas of flags
    iflags = np.argwhere(sr_flags.values)[:, 0]
    # iflag_groups = consecutive(iflags, 1)
    
    iflag_dates = sr_flags.index[iflags]
    dodgy_years = iflag_dates.year.unique()
    if 2014 in dodgy_years:
        dodgy_years = dodgy_years.drop(2014)
    
    fig, axes = plt.subplots(len(dodgy_years), 1,
                             figsize=[5, 3*len(dodgy_years)])
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for ax, year in zip(axes, dodgy_years):
        
        year_sr = sr.loc[sr.index.year==year]
        
        year_sr.plot(ax=ax, lw=.5)
        year_sr.resample('D').mean().plot(ax=ax, lw=1)
        
        year_flags = iflag_dates[iflag_dates.year==year]
        
        for date in year_flags:
            ax.axvspan(date, date + pd.Timedelta(days=1),
                       facecolor='red', alpha=.3, zorder=0)


        ax.set_title(year)
        
    plt.suptitle(f'{sr.name} flags={sr_flags.sum()}', y=.95)
    plt.subplots_adjust(hspace=.3)


#%%

def remove_consecutive_repeats(sr, thresh=8):
    new_sr = sr.copy()
    
    a = sr.values
    new_a = np.array([np.nan]*len(a))
    last = np.nan
    count = 0
    for i, v in enumerate(a):
        # print(v==last)

        if v == last:
            count += 1
            if count >= thresh:
                new_a[i-thresh:i] = np.nan
            else:
                new_a[i] = v
        else:
            count = 0
            new_a[i] = v
        last = v
        
    new_sr[:] = new_a
    
    return new_sr

#%% changepoint detection. this detects the presence of weird blocks 

def detect_daily_cv_changepoints(sr):
    
    # calculate difference between each hour of day
    hour_diff = sr.groupby(sr.index.hour).diff()
    
    # calculate variation coefficient of each day of hourly differences
    daily_cv = hour_diff.resample('D').std() / sr.resample('D').mean()
    # only keep values where >= 20 hours of the day were not NaN
    invalid = hour_diff.resample('D').count() >= 18
    daily_cv = daily_cv.where(invalid)
    
    # record days with 0 CV (exact day-to-day repeats)
    zero_cv = daily_cv == 0
    # remove from time series (as creates inf when log transforming)
    daily_cv = daily_cv.where(~zero_cv)
    
    # log transform daily cv
    daily_cv = np.log(daily_cv)
    
    algo = rpt.Window(width=30, model='rbf').fit(daily_cv.interpolate().dropna().values)
    result = algo.predict(pen=30)
    
    return len(result)-1
    

def get_changepoints(msr):

    algo = rpt.Pelt(model='rbf').fit(msr.bfill().ffill().values)
    result = algo.predict(pen=20)
    result = result[:-1]
    return len(result)-1


def get_length(sr):
    first, last = sr.dropna().iloc[[0, -1]].index
    length = len(pd.date_range(first, last, freq='H'))
    return length


#%% 

def get_flags(df):
    
    flags = {}
    cleaned = []
    for station in tqdm(df.columns):
        
        sr = df[station]
        
        # drop areas where there is less than 24 measurements in a week
        sr.loc[sr.rolling(24*7, center=True, min_periods=0).count() < 24] = np.nan
        
        possible_length = get_length(sr)
        actual_length = len(sr.dropna())
        
        # if less than half of data present, skip
        if actual_length/possible_length < .5:
            print('!')
            continue
        
        cleaned.append(remove_consecutive_repeats(remove_extremes(sr)))
        
        
        flags[station] = detect_daily_cv_changepoints(sr)
        print(station)
        
    flags = pd.Series(flags)
    cleaned = pd.concat(cleaned, axis=1)

    return flags, cleaned
# 
#%% main
for pol in ['no2', 'ozone', 'pm2_5', 'so2']:
    
    name_map = {'no2':'no2',
                'ozone':'o3',
                'pm2_5':'pm25',
                'so2':'so2'}
    
    # if os.path.exists(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/flags/{pol}_flags.csv'):
    #     print(f'{name_map[pol]} done!')
    #     continue
    
    # get flags
    df = open_cnemc_df(pol)
    flags, cleaned = get_flags(df)
    
    flags.to_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/flags/{name_map[pol]}_flags.csv')
    cleaned.to_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/cleaned/{name_map[pol]}_obs.csv')
