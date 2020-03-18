# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:43:36 2018

@author: ben
"""

# test the geo index!

from geo_index import geo_index
from geo_index import index_list_for_files
#from paths import ATL06_base

import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from ATL06_data import ATL06_data
from osgeo import osr
import sys

#SRS_proj4='+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
SRS_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
     
out_srs=osr.SpatialReference()
out_srs.ImportFromProj4(SRS_proj4)
ll_srs=osr.SpatialReference()
ll_srs.ImportFromEPSG(4326)
if hasattr(osr,'OAMS_TRADITIONAL_GIS_ORDER'):
  ll_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
ct=osr.CoordinateTransformation(ll_srs, out_srs) 


# NOTE: add your own path in paths.py
# (uncomment below to override)
ATL06_base='/home/suzanne/git_repos/ATL11/test_data
h5_files= glob(ATL06_base+'/*.h5')
resolution=[10000, 10000]

out_dir=ATL06_base+'/index/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    
REGEN_individual=True
REGEN_master=True

for h5_file in h5_files:    
    out_file=out_dir+'/'+h5_file.split('/')[-1]
    if os.path.isfile(out_file):
        if REGEN_individual:
            os.remove(out_file)
            geo_index(SRS_proj4=SRS_proj4, delta=resolution).from_list(index_list_for_files([h5_file], u'ATL06', resolution, SRS_proj4)).to_file(out_file)
    else:
        geo_index(SRS_proj4=SRS_proj4, delta=resolution).from_list(index_list_for_files([h5_file], u'ATL06', resolution, SRS_proj4)).to_file(out_file)

 
all_track_file=out_dir+'/'+'GeoIndex.h5'
#if REGEN:
if REGEN_master:
    if os.path.isfile(all_track_file):
        os.remove(all_track_file)   
    h5_list=glob(out_dir+'/*.h5')
    geo_index(SRS_proj4=SRS_proj4, delta=resolution).from_list(index_list_for_files(h5_list, u'h5_geoindex', resolution, SRS_proj4)).to_file(all_track_file)


#            
   
