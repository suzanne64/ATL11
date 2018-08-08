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

ATL06_base='/Volumes/ice2/nick/34_IceSat2/SyntheticTracks_Alex/ATL06/'
h5_files= glob(ATL06_base+'TrackData_*/*.h5')
out_top_dir='/Volumes/ice2/ben/Greenland_synthetic_track_index/';
if not os.path.isdir(out_top_dir):
    os.mkdir(out_top_dir)
REGEN=False

if REGEN:
    for h5_file in h5_files:    
        track_dir=h5_file.split('/')[-2]
        out_dir=out_top_dir+track_dir
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_file=out_dir+'/'+h5_file.split('/')[-1]
        print(out_file)
        if os.path.isfile(out_file):
            os.remove(out_file)
        geo_index().from_list(index_list_for_files([h5_file], u'ATL06', [1000, 1000], SRS_proj4)).to_file(out_file)
 

h5_list=glob('/Volumes/ice2/ben/Greenland_synthetic_track_index/TrackData_*/Track*.h5')
all_track_file=out_top_dir+'/'+'AllTracks.h5'
#if REGEN:
if REGEN:
    os.remove(all_track_file)
    geo_index().from_list(index_list_for_files(h5_list, u'h5_geoindex', [1000, 1000], SRS_proj4)).to_file(all_track_file)
    
geo_index().from_list(index_list_for_files(glob('/Volumes/ice2/ben/Greenland_synthetic_track_index/TrackData_*/Track_0605.h5'), u'h5_geoindex', [1000, 1000], SRS_proj4)).to_file(out_top_dir+'/Track605.h5') 

x0=np.arange(-100000, -90000, 1000)
y0=-2120000+np.arange(0, 20000, 1000)
x0, y0=np.meshgrid(x0, y0)
#GI_q=geo_index().from_file(all_track_file, read_file=False).query_xy(x0, y0)
plt.figure()
h5_605=glob('/Volumes/ice2/ben/Greenland_synthetic_track_index/TrackData_*/Track_0605.h5')
this_file=out_top_dir+'/Track605.h5'
xy_605=geo_index().from_file(this_file).bins_as_array()
#plt.plot(xy_605[:,0], xy_605[:,1],'ro')
 
#xy=geo_index().from_file(this_file, read_file=True).bins_as_array()
#plt.plot(xy[:,0], xy[:,1],'kx')
D6=geo_index().from_file(all_track_file, read_file=False).query_xy(x0, y0, delta=[1000,1000], get_data=True)
for D6_sub in D6:
    D6_sub.get_xy(SRS_proj4)
    plt.plot(D6_sub.x, D6_sub.y,'ro',markersize=12)
#  

x0=np.arange(-100000, -80000, 10000)
y0=-2120000+np.arange(0, 30000, 10000)
x0, y0=np.meshgrid(x0, y0)

#xy_605_10km=np.round(xy_605/10000)*10000
#ii=np.logical_and(np.in1d(xy_605_10km[:,0], x0), np.in1d(xy_605_10km[:,1], y0))
#plt.plot(xy_605[:,0], xy_605[:,1],'ro')
#plt.plot(xy_605[ii,0], xy_605[ii,1],'b.')


D6=geo_index().from_file(all_track_file, read_file=False).query_xy(x0, y0, delta=[10000,10000], get_data=True)
 
for D6_sub in D6:
    D6_sub.get_xy(SRS_proj4)
    plt.plot(D6_sub.x, D6_sub.y,'g.')
#plt.plot(x0, y0,'mp',markersize=14)
plt.axis('equal')

# Track 605 is the problem
# debug:
#h5_list=glob('/Volumes/ice2/ben/Greenland_synthetic_track_index/TrackData_01/Track*.h5')
#track_re=re.compile('Track_(\d+).h5')
#xyt=list()
#for h5 in h5_list:
#    xy=geo_index().from_file(h5, read_file=False).bins_as_array()
#    track_num=float(track_re.search(h5).group(1))
#    xyt.append(np.concatenate((xy, np.zeros((xy.shape[0],1))+track_num), axis=1))   
#xyt=np.concatenate(xyt, axis=0)
#plt.plot(xyt[:,0], xyt[:,1],'go');


#            
   