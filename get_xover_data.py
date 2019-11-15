#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:52:34 2019

@author: ben
"""

import numpy as np
from PointDatabase.point_data import point_data
from PointDatabase import geo_index

def get_xover_data(x0, y0, rgt, GI_files, xover_cache, index_bin_size, params_11):
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

    for x0_ctr in x0_ctrs:
        this_key=(np.real(x0_ctr), np.imag(x0_ctr))
        # check if we have already read in the data for this bin
        if this_key not in xover_cache:
            # if we haven't already read in the data, read it in.  These data will be in xover_cache[this_key]
            temp=[]
            for GI_file in GI_files:
                new_data = geo_index().from_file(GI_file).query_xy(this_key, fields=params_11.ATL06_xover_field_list);
                if new_data is not None:
                    temp += new_data
            if len(temp) == 0:
                xover_cache[this_key]=None
                continue
            xover_cache[this_key]={'D':point_data(list_of_fields=params_11.ATL06_xover_field_list).from_list(temp)}
            # remove the current rgt from data in the cache
            xover_cache[this_key]['D'].index(~np.in1d(xover_cache[this_key]['D'].rgt, [rgt]))
            if xover_cache[this_key]['D'].size==0:
                continue
            xover_cache[this_key]['D'].get_xy(EPSG=params_11.EPSG)
            # index the cache at 100-m resolution
            xover_cache[this_key]['index']=geo_index(delta=[100, 100], data=xover_cache[this_key]['D'])
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
                    D_xover.append(xover_cache[this_key]['D'].subset(np.arange(i0, i1+1, dtype=int)))
    if len(D_xover) > 0:
        D_xover=point_data().from_list(D_xover)
        
    # cleanup the cache if it is too large
    if len(xover_cache.keys()) > 50:
        cleanup_xover_cache(xover_cache, x0, y0, 2e4)
        
    return D_xover

def cleanup_xover_cache(cache, x0, y0, W):
    """
    delete entries in xover cache that are too far from the current point
    """
    for xy_bin in list(cache.keys()):
        if np.abs(x0+1j*y0 - (xy_bin[0]+1j*xy_bin[1])) > W:
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
