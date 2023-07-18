#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:30:38 2023

@author: eebjs
"""

import xarray as xr
import pandas as pd
from ufuncs import load_models
from constants import china_regions, pollutants, pretty_pol, common_grid
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


cams = xr.open_dataset('/nfs/a340/eebjs/wrf-mip/model_data/cams/regridded/cams_regridded.nc')

models = load_models()

#%%
def get_cams_diff(model):
    
    cams_masked = cams.where(model)
    
    diff = model - cams_masked
    return {'diff':diff, 'cams':cams_masked, 'model':model.where(cams_masked>0)}

diffs = {}
for name, model in models.items():
    print(name)
    diffs[name] = get_cams_diff(model)
    
#%% plot model diffs

# def plot_comparison_map(diff):

proj = ccrs.AzimuthalEquidistant(central_latitude=34.53333,
                                 central_longitude=109.83055)
latlon = ccrs.PlateCarree()
china_regions_gdf = china_regions.to_geodataframe().set_crs(latlon).to_crs(proj)

for name, diff in diffs.items():
    print(name)
    for pol in pollutants:
        print(pol)
        
        mod = diff['model'][pol].mean('time')
        cam = diff['cams'][pol].mean('time')
        mdiff = diff['diff'][pol].mean('time')
    
        pmup = max(float(mod.quantile(.99)),float(cam.quantile(.99)))
        diffabsmax = float(max(abs(mdiff.quantile(.02)), abs(mdiff.quantile(.98))))
        
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
        mod.plot(ax=axes[0], cbar_kwargs=cbar_kwargs,
                  transform=ccrs.PlateCarree(), zorder=2,
                  cmap='YlOrRd', vmin=0, vmax=pmup)
        axes[0].set_title(name)
            
        # diff
        mdiff.plot(ax=axes[1], cbar_kwargs=cbar_kwargs,
                  transform=ccrs.PlateCarree(), zorder=2,
                  vmin=-diffabsmax, vmax=diffabsmax,
                  cmap='RdBu_r')
        axes[1].set_title(f'{name} − CAMS',)
            
        # cams
        cam.plot(ax=axes[2], cbar_kwargs=cbar_kwargs,
                  transform=ccrs.PlateCarree(), zorder=2,
                  cmap='YlOrRd', vmin=0, vmax=pmup)
        axes[2].set_title('CAMS')
        
        fig.subplots_adjust(wspace=.02)
        fig.suptitle(pretty_pol[pol],y=.7)
        fig.savefig(f'/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/cams_comparison/{pol}_{name}.png', dpi=500, bbox_inches='tight')
        plt.close()

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
    
    for pol in pollutants:
        
    
        fig, axes = plt.subplots(2,3, figsize=[12,8])
        
        for ax, region in zip(np.append(axes.ravel()[:4], axes.ravel()[5]), region_names):
            mod = mask_by_region(diff['model'][pol], region)
            gwr = mask_by_region(diff['cams'][pol], region)
            
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
  
        fig.savefig(f'/nfs/see-fs-02_users/eebjs/acrobear/figs/wrf-mip/cams_comparison/bias_{pol}_{name}.png', dpi=500, bbox_inches='tight')
        
    