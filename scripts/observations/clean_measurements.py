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

bad = open_cnemc_year('no2', 2016)[1279]
good = open_cnemc_year('no2', 2017)[1275]
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
    
#%%

def flag_non_normal_variability(sr, plot=False):
    
    # calculate difference between each hour of day
    hour_diff = sr.groupby(sr.index.hour).diff()
    
    # calculate variation coefficient of each day of hourly differences
    daily_cv = hour_diff.resample('D').std() / sr.resample('D').mean()
    # only keep values where >= 20 hours of the day were not NaN
    invalid = hour_diff.resample('D').count() >= 20
    daily_cv = daily_cv.where(invalid)
    
    # record days with 0 CV (exact day-to-day repeats)
    zero_cv = daily_cv == 0
    # remove from time series (as creates inf when log transforming)
    daily_cv = daily_cv.where(~zero_cv).dropna()
    
    # log transform daily cv
    daily_cv = np.log(daily_cv)
    
    # fit normal distribution
    mu, std = stats.norm.fit(daily_cv)
    
    bins = np.linspace(daily_cv.quantile(.001), daily_cv.quantile(.999), 1000)
    bin_width = bins[1] - bins[0]
    bin_edges = np.linspace(bins[0] - bin_width/2, bins[-1]+bin_width/2, len(bins)+1)
    
    # Plot the histogram.
    actual_dens, _ = np.histogram(daily_cv, bins=bin_edges, density=True)
    normal_dens = stats.norm.pdf(bins, mu, std)
    
    if plot:
        plt.hist(daily_cv, bins=bins, density=True, color='green', alpha=.5)
        plt.plot(bins, normal_dens, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        xmin, xmax = plt.xlim()
        plt.title(title)
        
    # check how good fit is
    badness = pd.Series((actual_dens - normal_dens) / normal_dens)
    
    # flag bad data
    binned = pd.cut(daily_cv, bin_edges)
    badness.index = binned.value_counts().sort_index().index
    badness.loc[np.nan] = 0
    bseries = pd.Series(badness.loc[binned.values].values, index=binned.index)

    
    flags = bseries > 3
    flags = flags.resample('D').first()
    flags.loc[zero_cv] = True
    
    # rolling median flags to find areas of consecutive flags
    flags = flags.rolling(3, center=True).median()
    
    flags = flags.fillna(False)
   
    return flags.astype(int)
        

#%% 
df = open_cnemc_df('no2')
flags = []
for station in tqdm(df.columns):
    
    sr = df[station]
    
    if (sr.isnull().sum() / len(sr)) > .9:
        print('!')
        continue
    
    # first remove extreme values
    # sr = remove_extremes(df[station])

    sr = remove_consecutive_repeats(sr)    

    res = flag_non_normal_variability(sr, plot=False)
    res.name = station
    # print(res.sum())
    
    flags.append(res)
    
flags = pd.concat(flags, axis=1).fillna(0)

#%%

#%% calculate flags/day

flag_prop = pd.Series(index=df.columns, dtype=float)
for station in df.columns:

    
    sr = df[station]
    days_with_data = (sr.resample('D').count() >= 20).sum()
    
    if days_with_data < 365:
        continue
    
    flagged_days = flags[station].sum()

    flag_prop.loc[station] =  flagged_days / days_with_data
    
for station_id in flag_prop.nlargest(20).index:

    
    sr = df[station_id]
    sr_flags = flags[station_id]
    
    plot_flags(sr, sr_flags)
    
    