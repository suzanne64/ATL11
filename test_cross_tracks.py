# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:36:51 2018

@author: ben
"""
from geo_index import geo_index
from geo_index import index_list_for_files
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from ATL06_data import ATL06_data
from osgeo import osr
import re
from ATL11_data import ATL11_data
from geo_index import unique_points
from geo_index import get_data_for_geo_index



SRS_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
index_top_dir='/Volumes/ice2/ben/Greenland_synthetic_track_index/'
all_track_file=index_top_dir+'/'+'AllTracks.h5'

ATL11_file='ATL11_h5/ATL11_Track605_Pair2.h5'

D11=ATL11_data().from_file(ATL11_file)


# transform the D11 points into an xy coordinate system:
xy_srs=osr.SpatialReference()
xy_srs.ImportFromProj4( SRS_proj4)
ll_srs=osr.SpatialReference()
ll_srs.ImportFromEPSG(4326)
ct=osr.CoordinateTransformation(ll_srs, xy_srs)
x, y, z = list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(D11.corrected_h.ref_pt_lon), np.ravel(D11.corrected_h.ref_pt_lat), np.zeros_like(np.ravel(D11.corrected_h.ref_pt_lat)))]))
 
xb,yb=unique_points(x,y, [1000., 1000.])

query_out=geo_index().from_file(all_track_file, read_file=False).query_xy([xb, yb], delta=[1000, 1000], pad=1, get_data=False)
plt.figure(); 
plt.plot(xb,yb,'k.')

track_re=re.compile('Track_(\d+)')

for qq in query_out:
    if track_re.search(qq).group(1)=='0605':
        continue
    plt.plot(query_out[qq]['x'], query_out[qq]['y'],'bx')
    D6_list=get_data_for_geo_index({qq:query_out[qq]},fields={None:('latitude','longitude','delta_time','h_li')})
    for D6 in get_data_for_geo_index({qq:query_out[qq]},fields={None:('latitude','longitude','delta_time','h_li')}):
        D6.get_xy(SRS_proj4)
        plt.plot(D6.x, D6.y,'.')
plt.axis('equal')

 

