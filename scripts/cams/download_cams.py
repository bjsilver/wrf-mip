#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:38:39 2023

@author: eebjs
"""
import cdsapi
import yaml

with open('/nfs/see-fs-02_users/eebjs/.adsapirc', 'r') as f:
        credentials = yaml.safe_load(f)
c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'date': '2017-05-01/2019-09-01',
        'format': 'netcdf',
        'variable': [
            'nitrogen_dioxide', 'ozone',
            'particulate_matter_2.5um', 'sulphur_dioxide',
        ],
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'area': [78, 70, 21,
            152,
            
        ],
        'model_level': '60'
    },
    
    '/nfs/a340/eebjs/wrf-mip/model_data/CAMS_reanalysis.nc')