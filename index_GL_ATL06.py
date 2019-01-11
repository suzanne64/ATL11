# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:43:36 2018

@author: ben
"""

# test the geo index!

from geo_index import geo_index
from geo_index import index_list_for_files
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from ATL06_data import ATL06_data
from osgeo import osr
import re

SRS_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
     
out_srs=osr.SpatialReference()
out_srs.ImportFromProj4(SRS_proj4)
ll_srs=osr.SpatialReference()
ll_srs.ImportFromEPSG(4326)
ct=osr.CoordinateTransformation(ll_srs, out_srs) 

ATL06_base='/Volumes/ice2/ben/scf/GL_06/ASAS/944'
h5_files= glob(ATL06_base+'/*.h5')
resolution=[10000, 10000]


#h5_files=glob('/Volumes/ice2/ben/scf/GL_06/ASAS/944/ATL06_20181015140742_02590104_944_01.h5')

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
        geo_index(SRS_proj4=SRS_proj4).from_list(index_list_for_files([h5_file], u'ATL06', resolution, SRS_proj4)).to_file(out_file)
    else:
        geo_index(SRS_proj4=SRS_proj4).from_list(index_list_for_files([h5_file], u'ATL06', resolution, SRS_proj4)).to_file(out_file)

h5_list=glob(out_dir+'/*.h5')
all_track_file=out_dir+'/'+'GeoIndex.h5'
#if REGEN:
if REGEN_master:
    if os.path.isfile(all_track_file):
        os.remove(all_track_file)   
    geo_index(SRS_proj4=SRS_proj4).from_list(index_list_for_files(h5_list, u'h5_geoindex', resolution, SRS_proj4)).to_file(all_track_file)


#            
   