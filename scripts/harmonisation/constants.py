#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:30 2023

@author: eebjs
"""

import xesmf as xe

# create the common grid
common_grid = xe.util.grid_2d(67, 140, .1, 16, 56, .1)

