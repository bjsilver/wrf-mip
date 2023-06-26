#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:30:38 2023

@author: eebjs
"""

import xarray as xr
import pandas as pd
from ufuncs import load_models
from constants import china_regions
import xesmf as xe
import numpy as np
import regionmask
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
import matplotlib.colors as colors
import matplotlib.cm as cmaps
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_vandonk_year(year):
    vdpath = '/nfs/a340/eebjs/acrobear/vandonk/p1deg/Monthly/'
    year = str(year)

    das = []
    for month in range(1,13):
        month = str(month).zfill(2)
        ds = xr.open_dataset(vdpath+\
                             f'V5GL02.HybridPM25c_0p10.Global.{year}{month}-{year}{month}.nc')
        da = ds['GWRPM25']

        # drop empty lat
        da = da.dropna(dim='lat', how='all')

        das.append(da)

    da = xr.concat(das, dim='mon')
    datetime = pd.date_range(start=year+'-01-01', end=year+'-12-31',
                             freq='MS')
    da = da.assign_coords({'mon':datetime})
    da = da.rename({'mon':'time'})

    return da

vd = load_vandonk_year(2017)
# slice to JJA
vd = vd.loc[{'time':slice('2017-06-01', '2017-08-30')}]
# slice to China area
vd = vd.loc[{'lat':slice(15.5, 56.5), 'lon':slice(66.5, 140.5)}]

models = load_models()

#%% regrid
common_grid = xr.Dataset(coords={'lon':np.arange(67, 140, .1), 'lat':np.arange(16,56,.1)})

regridder = xe.Regridder(vd, common_grid, method='bilinear')
vd_regridded = regridder(vd)

#%%
def get_vd_diff(model):
    mean_model = model['pm25'].resample({'time':'MS'}).mean()
    
    vd_masked = vd_regridded.where(mean_model)
    
    diff = mean_model - vd_masked
    return {'diff':diff, 'vd':vd_masked, 'model':mean_model.where(vd_masked>0)}

diffs = {}
for name, model in models.items():  
    diffs[name] = get_vd_diff(model)
    
#%% plot model diffs

# def plot_comparison_map(diff):

proj = ccrs.AzimuthalEquidistant(central_latitude=34.53333,
                                 central_longitude=109.83055)
latlon = ccrs.PlateCarree()
china_regions_gdf = china_regions.to_geodataframe().set_crs(latlon).to_crs(proj)

for name, diff in diffs.items():
    
    pmup = float(max(diff['model'].quantile(.98),
               diff['vd'].quantile(.98)))
    diffabsmax = max(abs(diff['diff'].quantile(.05)), abs(diff['diff'].quantile(.95)))
    
    fig, axes = plt.subplots(1,3, subplot_kw={'projection':proj},
                             figsize=[12,6])
    
    for ax in axes:
        ax.set_extent([75, 133, 17, 53], crs=latlon)
        ax.add_feature(cf.OCEAN, zorder=0)
        ax.add_feature(cf.LAND, zorder=1)
        # ax.add_feature(cf.BORDERS)
        ax.coastlines(lw=.2, zorder=3)
        china_regions_gdf.plot(ax=ax, facecolor='none', edgecolor='black', lw=.5, zorder=4)

    cbar_kwargs = {'shrink':.8, 'orientation':'horizontal',
                   'pad':.03, 'label':'$\mathrm{\mu g\ m^{-3}}$'}
    
    # model
    diff['model'].mean('time')\
        .plot(ax=axes[0], cbar_kwargs=cbar_kwargs,
              transform=ccrs.PlateCarree(), zorder=2,
              cmap='YlOrRd', vmin=0, vmax=pmup)
    axes[0].set_title(name)
        
    # diff
    diff['diff'].mean('time')\
        .plot(ax=axes[1], cbar_kwargs=cbar_kwargs,
              transform=ccrs.PlateCarree(), zorder=2,
              vmin=-diffabsmax, vmax=diffabsmax,
              cmap='RdBu_r')
    axes[1].set_title(f'{name} − GWR',)
        
    # vandonk
    diff['vd'].mean('time')\
        .plot(ax=axes[2], cbar_kwargs=cbar_kwargs,
              transform=ccrs.PlateCarree(), zorder=2,
              cmap='YlOrRd', vmin=0, vmax=pmup)
    axes[2].set_title('GWR')
    
    fig.subplots_adjust(wspace=.02)
    fig.savefig(f'/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/vandonk_comparison/{name}.png', dpi=500, bbox_inches='tight')
    
#%% 
china_mask = china_regions.mask(common_grid)
region_names = china_regions.names
def mask_by_region(da, name):
    mask_number = china_regions.map_keys(name)
    masked = da.mean('time').where(china_mask==mask_number)
    return masked

region_colors =    {'Western China':'#cfba82',
                    'Northern East China':'orangered',
                    'Northeast China':'#3ac7c9',
                    'Southwest China':'#018a28',
                    'Southern East China':'#6b3a9c'}
for region in china_regions_gdf.index:
    china_regions_gdf.loc[region, 'color'] = region_colors[china_regions_gdf.loc[region, 'names']]

for name, diff in diffs.items():
    
    fig, axes = plt.subplots(2,3, figsize=[12,8])
    
    for ax, region in zip(np.append(axes.ravel()[:4], axes.ravel()[5]), region_names):
        mod = mask_by_region(diff['model'], region)
        gwr = mask_by_region(diff['vd'], region)
        
        topbin = float(((mod.quantile(.99)+gwr.quantile(.99))/2))
        
        mod_binned = mod.groupby_bins(gwr, bins=np.linspace(1,topbin, 101)).mean()
        vd_binned = gwr.groupby_bins(gwr, bins=np.linspace(1,topbin, 101)).mean()
        
        # maxbin = float(max(mod_binned.max(), vd_binned.max()))

        sr = (mod_binned-vd_binned).to_pandas()
        sr.index = [(i.left+i.right)/2 for i in sr.index]

        
        x = sr.index
        y = sr.values
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        cbar_bound = max(abs(sr.max()), abs(sr.min()))
        # norm = plt.Normalize(-cbar_bound, cbar_bound)
        norm = plt.Normalize(-50,50)
        lc = LineCollection(segments, cmap='RdBu_r', norm=norm)
        # Set the values used for colormapping
        lc.set_array(y)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        ax.set_ylim(sr.min()-cbar_bound*.03,sr.max()+cbar_bound*.03)
        ax.set_xlim(0,100)
        ax.set_facecolor('grey')
        for spine in ax.spines.values():
            spine.set_edgecolor(region_colors[region])
            spine.set_linewidth(3)
        
        ax.set_title(region)
        
        ax.axhline(0, color='white', lw=2)
        ax.grid(color='white')
        ax.set_xlabel('concentration ($\mathrm{\mu g\ m^{-3}}$)')
        ax.set_ylabel('bias ($\mathrm{\mu g\ m^{-3}}$)')
        
    
    axes[1,1].remove()
    mapax = fig.add_subplot(2, 3, 5, projection=proj)
    
    mapax.set_extent([75, 133, 17, 53], crs=latlon)
    mapax.add_feature(cf.OCEAN, zorder=0)
    mapax.add_feature(cf.LAND, zorder=1)
    # ax.add_feature(cf.BORDERS)
    mapax.coastlines(lw=.2, zorder=3)
    china_regions_gdf\
        .plot(ax=mapax, edgecolor='black',
              lw=.5, zorder=4, color=china_regions_gdf['color'])

    cbar_ax = fig.add_axes([0.2, 0, 0.6, 0.04])
    fig.colorbar(line, cax=cbar_ax, orientation='horizontal',
                 label='bias ($\mathrm{\mu g\ m^{-3}}$)')
    fig.suptitle(name, fontsize=22, y=.1)
    fig.subplots_adjust(wspace=.3, hspace=.3)
    fig.savefig(f'/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/vandonk_comparison/bias_{name}.png', dpi=500, bbox_inches='tight')
    
    