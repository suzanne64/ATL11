#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:52:34 2019

@author: ben
"""

import numpy as np
#from PointDatabase.point_data import point_data
#from PointDatabase import geo_index
import pointCollection as pc
import h5py
import re
from ATL11 import apply_release_bias
from ATL11.check_ATL06_hold_list import check_ATL06_data_against_hold_list

def get_ATL06_release(D6):
    if D6 is None:
        return
    #Get the release number for ATL06 data from the tile files
    for D6i in D6:
        if D6i is None:
            continue
        u_file, i_file = np.unique(D6i.source_file_num, return_inverse=True)
        u_release=np.zeros_like(u_file)
        ATL06_re = re.compile('ATL06_\d+_\d+_(\d{3})_\d{2}.h5')
        with h5py.File(D6i.filename) as h5f:
            for ii, file_num in enumerate(u_file):
                source_file=h5f['source_files'].attrs[f'file_{int(file_num)}']
                u_release[ii]=int(ATL06_re.search(source_file).group(1))
        release = u_release[i_file]
        release.shape = D6i.h_li.shape
        D6i.assign({'release' : release})

def get_xover_data(x0, y0, rgt, GI_files, xover_cache, index_bin_size, params_11,
                   release_bias_dict=None,
                   hold_list=None,
                   verbose=False, xy_bin=None):
    """
    Read the data from other tracks.

    Maintain a cache of data so that subsequent reads don't have to reload data from disk
    Inputs:
        x0, y0: bin centers
        rgt: current rgt
        GI_files: lsti of geograpic index file
        xover_cache: data cache (dict)
        index_bin_size: size of the bins in the index
        params_11: default parameter values for the ATL11 fit
    """

    # identify the crossover centers
    x0_ctrs = buffered_bins(x0, y0, 2*params_11.L_search_XT, index_bin_size)
    D_xover=[]
    ATL06_fields = ['delta_time', 'latitude','longitude', 'h_li', 'h_li_sigma',
                    'atl06_quality_summary', 'segment_id', 'sigma_geo_h',
                    'x_atc', 'y_atc', 'dh_fit_dx',
                    'sigma_geo_at','sigma_geo_xt', 'sigma_geo_r',
                    'ref_azimuth', 'ref_coelv',
                    'tide_ocean', 'dac',
                    'rgt', 'cycle_number',
                    'BP',  'spot', 'LR',
                    'source_file_num']

    for x0_ctr in x0_ctrs:
        this_key=(np.real(x0_ctr), np.imag(x0_ctr))
        # check if we have already read in the data for this bin
        if this_key not in xover_cache:
            if verbose > 1:
                print(f"reading {this_key}")
            # if we haven't already read in the data, read it in.  These data will be in xover_cache[this_key]
            temp=[]
            for GI_file in GI_files:
                new_data = pc.geoIndex().from_file(GI_file).query_xy(this_key, fields=ATL06_fields)
                get_ATL06_release(new_data)
                if new_data is None:
                    continue
                if xy_bin is not None:
                    subset_data_to_bins(new_data, xy_bin, EPSG=params_11.EPSG)
                temp += new_data
            if len(temp) == 0:
                xover_cache[this_key]=None
                continue
            temp=pc.data(fields=params_11.ATL06_xover_field_list + ['release']).from_list(temp)
            if release_bias_dict is not None:
                apply_release_bias(temp, release_bias_dict)
            xover_cache[this_key]={'D':temp}
            # remove the current rgt from data in the cache
            temp.index(~np.in1d(xover_cache[this_key]['D'].rgt, [rgt]))
            if xover_cache[this_key]['D'].size==0:
                continue
            temp.get_xy(EPSG=params_11.EPSG)
            sort_data_bin(temp, 100)
            xover_cache[this_key]['D']=temp
            # index the cache at 100-m resolution
            xover_cache[this_key]['index']=pc.geoIndex(delta=[100, 100], data=xover_cache[this_key]['D'])
        # now read the data from the crossover cache
        if (xover_cache[this_key] is not None) and (xover_cache[this_key]['D'] is not None):
            try:
                Q=xover_cache[this_key]['index'].query_xy([x0, y0], pad=1, get_data=False)
            except KeyError:
                Q=None
            if Q is None:
                continue
            # if we have read in any data for the current bin, subset it to the bins around the reference point
            for key in Q:
                for i0, i1 in zip(Q[key]['offset_start'], Q[key]['offset_end']):
                    D_xover.append(xover_cache[this_key]['D'][np.arange(i0, i1+1, dtype=int)])
    if len(D_xover) > 0:
        D_xover=pc.data().from_list(D_xover)
        if hold_list is not None:
            check_ATL06_data_against_hold_list(D_xover, hold_list)

    # cleanup the cache if it is too large
    if len(xover_cache.keys()) > 5:
        cleanup_xover_cache(xover_cache, x0, y0, 2e4, verbose=verbose)

    return D_xover

def subset_data_to_bins(D, xy_bin, bin_size=100, EPSG=None):
    if D is None:
        return
    for Di in D:
        Di.get_xy(EPSG=EPSG)
        Di.index(np.in1d(np.round((Di.x+1j*Di.y)/bin_size)*bin_size, xy_bin[:,0]+1j*xy_bin[:,1]))

def sort_data_bin(D, bin_W):
    """
    Sort the entries of a data structure so that they get indexed in contiguous chunks
    """
    y_bin_function=np.round(D.y/bin_W)
    x_bin_function=np.round(D.x/bin_W)
    x_scale=np.nanmax(x_bin_function)-np.nanmin(x_bin_function)
    t=D.delta_time
    t_scale=np.nanmax(t)-np.nanmin(t)
    xy_bin_function=(y_bin_function-np.nanmin(y_bin_function))*x_scale+(x_bin_function-np.nanmin(x_bin_function))
    xyt_bin_function= xy_bin_function + (t-np.nanmin(t))/t_scale
    ind=np.argsort(xyt_bin_function)
    return D.index(ind)


def cleanup_xover_cache(cache, x0, y0, W, verbose=False):
    """
    delete entries in xover cache that are too far from the current point
    """
    for xy_bin in list(cache.keys()):
        if np.abs(x0+1j*y0 - (xy_bin[0]+1j*xy_bin[1])) > W:
            if verbose > 1:
                print(f"cleaning up {xy_bin}")
            del(cache[xy_bin])


def buffered_bins(x0, y0, w_buffer, w_bin, complex=True):
    """
    Generate a set of bins that enclose a set of points, including a buffer
    """
    dx, dy=np.meshgrid([-w_buffer, 0, w_buffer], [-w_buffer, 0, w_buffer])
    dx.shape=[9, 1];
    dy.shape=[9, 1]
    xr=np.unique(np.round(x0/w_buffer)*w_buffer+1j*np.round(y0/w_buffer)*w_buffer)
    xr=np.unique(xr.ravel()+dx+1j*dy)
    xr=np.unique(np.round(xr/w_bin)*w_bin)
    if complex:
        return xr
    else:
        return np.real(xr), np.imag(xr)
