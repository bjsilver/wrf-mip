#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:18:00 2023

@author: ben
"""

import pandas as pd
import matplotlib.pyplot as plt
from constants import models, pollutants, pretty_model, pretty_pol, china_regions
import matplotlib.dates as mdates
import numpy as np
import copy
from scipy.stats import pearsonr
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cf

measurement_path = '/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/'
meta = pd.read_csv(measurement_path+'metadata_with_region.csv', index_col=0)
meta.index = meta.index.astype(str)


spath = '/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/cnemc_comparison/'

def load_timeseries(drop_flagged=True):

    # load measurements
    cnemc = {}
    model_dict = {}
    for pol in pollutants:
        print(pol)
        
        meas = pd.read_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/cleaned/{pol}_obs.csv', 
                              index_col=0, parse_dates=True)
        if drop_flagged:
            flagged = meta[f'{pol}_clean']
            flagged = flagged[flagged]
            meas = meas[[i for i in meas if i in flagged.index]]
            
        # slice to JJA 2017
        meas = meas.loc[slice('2017-06-01', '2017-08-31')]
        # drop empty time series
        meas = meas.dropna(axis=1, how='all')
        # drop where less than 75% of data is present
        long = meas.isna().sum() < len(meas)/4
        long = long[long]
        meas = meas[long.index]
        meas = meas.tz_convert('Asia/Shanghai')
        
        cnemc[pol] = meas
        
        model_dfs = {}
        for model in models:
            model_df = pd.read_csv(f'/nfs/see-fs-02_users/eebjs/wrf-mip/data/cnemc_measurements/{model}_interped/{pol}_interped.csv', 
                                  index_col=0, parse_dates=True)
            model_df = model_df[[i for i in model_df if i in meas]]
            # convert timezone
            model_df = model_df.tz_convert('Asia/Shanghai').loc[slice('2017-06-01', '2017-08-31')]
            model_dfs[model] = model_df
            
        model_dict[pol] = model_dfs
            
    return {'obs':cnemc, 'mod':model_dict}

d = load_timeseries(drop_flagged=True)

#%% plotting functions

def daily_plot(d, sname, title=None):
    
    obs = d['obs']
    mod = d['mod']
    
    fig, axes = plt.subplots(2,2, figsize=[10,6.5])
    
    for pol, ax in zip(pollutants, axes.ravel()):
        
        obs[pol].resample('D').mean().mean(1)\
            .plot(ax=ax, color='black', label='CNEMC')
        for model, modf in mod[pol].items():
            
            modf.resample('D').mean().mean(1)\
                .plot(ax=ax, label=pretty_model[model],
                      legend=None)      
                
        ax.set_title(pretty_pol[pol])
        ax.set_ylabel('µg m⁻³')
        ax.set_xlabel('')
        ax.set_xticks(pd.date_range('2017-06-01', '2017-08-31', freq='MS'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(alpha=.4)
        
    fig.legend(*ax.get_legend_handles_labels(), ncol=3,
               bbox_to_anchor=[.666, 0.07])
    fig.subplots_adjust(hspace=.3)
    fig.suptitle(title, fontsize=16, y=.96)
    fig.savefig(spath+f'{sname}.svg', bbox_inches='tight')
    


def diurnal_plot(d, sname, title=None):
    
    obs = d['obs']
    mod = d['mod']
    
    fig, axes = plt.subplots(2,2, figsize=[10,6.5])
    
    for pol, ax in zip(pollutants, axes.ravel()):
        
        obs[pol].groupby(obs[pol].index.hour).mean().mean(1)\
            .plot(ax=ax, color='black', label='CNEMC')
        for model, modf in mod[pol].items():

            modf.groupby(modf.index.hour).mean().mean(1)\
                .plot(ax=ax, label=pretty_model[model],
                      legend=None)        
                
        ax.set_title(pretty_pol[pol])
        ax.set_ylabel('µg m⁻³')
        ax.set_xlabel('')
        ax.grid(alpha=.4)
        ax.set_xticks(np.arange(0,24,4))
        ax.margins(x=0)
        ax.set_xlim(0,24)
        
    fig.legend(*ax.get_legend_handles_labels(), ncol=3,
               bbox_to_anchor=[.666, 0.07])
    fig.subplots_adjust(hspace=.3)
    fig.suptitle(title, fontsize=16)
    fig.savefig(spath+f'{sname}.svg', bbox_inches='tight')
    

#%% plots for all China
daily_plot(d, sname='daily_mean_comparison')
diurnal_plot(d, sname='diurnal_comparison')

#%% plots by region

def stations_by_region(d, region):
    
    dcop = copy.deepcopy(d)
    
    in_region = meta.index[meta.region == region]
    
    obs = dcop['obs']
    mod = dcop['mod']
    
    for pol in pollutants:
        df = obs[pol]
        df_in =df[[i for i in df.columns if i in in_region]]
        obs[pol] = df_in
        
        for model, modf in mod[pol].items():
            df_in = modf[[i for i in modf.columns if i in in_region]]
            mod[pol][model] = df_in
            
    return {'obs':obs, 'mod':mod}
        

for region in china_regions.names:
    print(region)
    daily_plot(stations_by_region(d, region), sname=f'daily_mean_comparison_{region}',
               title=region)
    diurnal_plot(stations_by_region(d, region), sname=f'diurnal_comparison_{region}',
                 title=region)

#%% R2 and NMB plots

def get_metrics(d, pol, resample_freq):
    
    dcop = copy.deepcopy(d)
    obs = dcop['obs'][pol].resample(resample_freq).mean()
    mod = dcop['mod'][pol]
    
    modelstats = {}
    for model, modf in mod.items():
        
        modf = modf.resample(resample_freq).mean()
        shared = modf.columns.intersection(obs.columns)
        
        stats = pd.DataFrame(index=shared, columns=['r', 'nmb'])
        for station in tqdm(shared):
            
            m = modf[station].dropna()
            o = obs[station].dropna()
            tindex = m.index.intersection(o.index)
            
            stats.loc[station, 'r'] = pearsonr(m.loc[tindex], o.loc[tindex]).statistic
            stats.loc[station, 'nmb'] = (m - o).sum() / o.sum()
            
        modelstats[model] = stats
        
    return modelstats



markers = ['o', '^', 's']
model_markers = dict((k, v) for k, v in zip(models, markers))

china_regions_gdf = china_regions.to_geodataframe()

region_colors =    {'Western China':'#cfba82',
                    'Northern East China':'orangered',
                    'Northeast China':'#3ac7c9',
                    'Southwest China':'#018a28',
                    'Southern East China':'#6b3a9c'}

for region in china_regions_gdf.index:
    china_regions_gdf.loc[region, 'color'] = region_colors[china_regions_gdf.loc[region, 'names']]


def metrics_plot(pmetrics, sname):
    
    def plot_dist(r, nmb, color, ax):
        x = r.median()
        y = nmb.median()
        
        ax.scatter(r, nmb, color=color, marker = model_markers[model],
                   label=None, zorder=500, s=1, alpha=.1)
            
        # point
        ax.scatter(x, y, color=color, marker = model_markers[model],
                   label=pretty_model[model], zorder=10000, s=100,
                   alpha=1, edgecolor='white')

    fig, axes = plt.subplots(2,2, figsize=[10,10])
    for ax, pol in zip(axes.ravel(), pollutants):
    
        metrics = pmetrics[pol]
    
        for model in models:
            # plot for all country average
            plot_dist(r=metrics[model]['r'].astype(float), 
                      nmb=metrics[model]['nmb'].astype(float), 
                      color='black', ax=ax)
            
            for region in china_regions.names:
                in_region = meta.index[meta.region == region]
                
                rmet = metrics[model].loc[in_region.intersection(metrics[model].index)]
                plot_dist(r=rmet['r'].astype(float), 
                          nmb=rmet['nmb'].astype(float), 
                          color=region_colors[region], ax=ax)
            
            
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('r')
        ax.set_ylabel('NMB')
        ax.axhline(0, color='black', lw=.7)
        ax.grid(color='black', alpha=.2)
        ax.set_facecolor('lightgrey')
    
        
        ax.set_title(pretty_pol[pol])
        
    
    proj = ccrs.AzimuthalEquidistant(central_latitude=34.53333,
                                     central_longitude=109.83055)
    mapax = fig.add_axes([.93, .34, .15, .15], projection=proj)
    
    
    mapax.coastlines(lw=.2, zorder=3)
    china_regions_gdf\
        .plot(ax=mapax, edgecolor='black',
              lw=.2, zorder=4, color=china_regions_gdf['color'])
    mapax.axis('off')
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::6], labels[::6], bbox_to_anchor=[1.07, 0.62])
    fig.subplots_adjust(hspace=.3, wspace=.3)
    
    fig.savefig(spath+sname, bbox_inches='tight')

#%% daily

pmetrics = {}
for pol in pollutants:
    pmetrics[pol] = get_metrics(d, pol, resample_freq='D')
metrics_plot(pmetrics, sname='daily_model_performance.svg')

#%% hourly

pmetrics = {}
for pol in pollutants:
    pmetrics[pol] = get_metrics(d, pol, resample_freq='H')
metrics_plot(pmetrics, sname='hourly_model_performance.svg')



#%%

proj = ccrs.AzimuthalEquidistant(central_latitude=34.53333,
                                 central_longitude=109.83055)
latlon = ccrs.PlateCarree()
china_regions_gdf = china_regions.to_geodataframe().set_crs(latlon).to_crs(proj)
outer = china_regions_gdf.dissolve()

#%%


for pol in pollutants:
    print(pol)
    
    obs = d['obs'][pol].mean().dropna()
    pmod = d['mod'][pol]
    
    for name in models:
    
        fig, axes = plt.subplots(1,3, subplot_kw={'projection':proj},
                                 figsize=[12,6])
        
        mod = pmod[name].mean().dropna()
        
        shared = mod.index.intersection(obs.index)
        obs = obs.loc[shared]
        mod = mod.loc[shared]
        
        dif = mod - obs
        
        pmup = max(float(mod.quantile(.99)),float(obs.quantile(.99)))
        difflim = max(abs(dif.quantile(.95)), abs(dif.quantile(.05)))
        
        
        
        for ax in axes:
            ax.set_extent([75, 133, 17, 53], crs=latlon)
            ax.add_feature(cf.OCEAN, zorder=0)
            ax.add_feature(cf.LAND, zorder=1, color='grey')
            ax.add_feature(cf.BORDERS, color='white', lw=.5)
            # ax.coastlines(lw=.2, zorder=3)
            # outer.plot(ax=ax, color='black', edgecolor='white', lw=1, zorder=4)
            # outer.geometry.boundary.plot(ax=ax)
       
        
        # model

        sc = \
        axes[0].scatter(meta.loc[shared, 'lon'], meta.loc[shared, 'lat'],
                        cmap='YlOrRd', vmin=0, vmax=pmup, c=mod,
                        transform=latlon, s=15, alpha=.6)
        cb = plt.colorbar(sc, orientation='horizontal', pad=.03, 
                     label='$\mathrm{\mu g\ m^{-3}}$', alpha=1)
        cb.set_alpha(1)
        cb.draw_all()
        axes[0].set_title(pretty_model[name])
        
            
        # diff
        sc = \
        axes[1].scatter(meta.loc[shared, 'lon'], meta.loc[shared, 'lat'],
                        cmap='RdBu_r', vmin=-difflim, vmax=difflim, c=dif,
                        transform=latlon, s=15, alpha=.6)
        cb = plt.colorbar(sc, orientation='horizontal', pad=.03, 
                     label='$\mathrm{\mu g\ m^{-3}}$', alpha=1)
        cb.set_alpha(1)
        cb.draw_all()
        axes[1].set_title(f'{pretty_model[name]} − CNEMC')
            
        # obs
        sc = \
        axes[2].scatter(meta.loc[shared, 'lon'], meta.loc[shared, 'lat'],
                        cmap='YlOrRd', vmin=0, vmax=pmup, c=obs,
                        transform=latlon, s=15, alpha=.6)
        cb = plt.colorbar(sc, orientation='horizontal', pad=.03, 
                     label='$\mathrm{\mu g\ m^{-3}}$', alpha=1)
        cb.set_alpha(1)
        cb.draw_all()
        axes[2].set_title('CNEMC')

        fig.suptitle(pretty_pol[pol], y=.68)
        fig.savefig(f'/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/cnemc_comparison/{pol}_{name}.png', dpi=500, bbox_inches='tight')
        

        plt.close()

