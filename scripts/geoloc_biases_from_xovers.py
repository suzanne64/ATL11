#! /usr/bin/env python

import pointCollection as pc
import numpy as np
import glob
import h5py
import re
import scipy.sparse as sp
import ATL11
import pandas as pd
import argparse
from ATL11.check_ATL06_hold_list import read_hold_files    

def RDE(x):
    xs=x.copy()
    xs=np.isfinite(xs)   # this changes xs from values to a boolean
    if np.sum(xs)<2 :
        return np.nan
    ind=np.arange(0.5, np.sum(xs))
    LH=np.interp(np.array([0.16, 0.84])*np.sum(xs), ind, np.sort(x[xs]))
    #print('LH =',LH)
    return (LH[1]-LH[0])/2.  # trying to get some kind of a width of the data ~variance

def read_xovers(file, fields=None, get_data=False):
    '''
    Read crossovers from a saved crossover file
    
    Inputs :
    File (str): hdf-5 file to read
    fields (list of strs): fields to read from the file
    
    Returns:
    v (pc.data) field values interpolated to the crossover location
    m (pc.data) Metadata fields for each crossover location, including grounded status, slope, and location
    data (list of pc.data objects) raw data read from each crossover file
    '''
    
    if fields is None:
        fields=['x','y','delta_time','h_li','h_li_sigma','h_mean','spot', 'rgt', 
                'dh_fit_dx','dh_fit_dy','atl06_quality_summary', 'latitude',
                'seg_azimuth','ref_azimuth','ref_coelv', 'cycle_number']
    
    m = pc.data().from_h5(file, field_dict={None:['grounded','x','y','slope_x','slope_y']})
    v=pc.data(columns=2)
    data = [pc.data(columns=2).from_h5(file, field_dict={group:fields+['W']}) for group in ['data_0','data_1']]
    d=pc.data()
    for field in fields:
        if field=='W':
            continue
        temp = np.zeros((m.size, 2))
        for col in [0, 1]:
            temp[:, col] = np.sum(data[col].W * getattr(data[col], field), axis=1)
        
        if field in ['rgt','cycle','spot']:
            # integer fields:
            v.assign({ field : np.round(temp).astype(int)})
        else:
            v.assign({ field : temp})
            
    for item in [v, d]:
        item.__update_size_and_shape__()

    return v, m, data

def filter_xovers(v, m, data, 
                  slope_max = 0.2, grounded_tol = 0.99, 
                max_delta_t = None, 
                  min_h=None):
    '''
    Filter the crossovers based on field values
    '''

    good=np.all(v.atl06_quality_summary < 0.01, axis=1)
    for di in data:
        good &= np.abs(di.delta_time[:,1]-di.delta_time[:,0]) < 0.005
    good &= m.grounded > grounded_tol
    if slope_max is not None:
        good &= np.all(np.abs(np.c_[v.dh_fit_dx, v.dh_fit_dy])< slope_max, axis=1)
    
    if max_delta_t is not None:
        good &= (v.delta_t[:,1]-v.delta_t[:,0]) < max_delta_t
    if min_h is not None:
        min_h &= (np.all(v.h_li > min_h, axis=1))
    
    v.index(good)
    m.index(good)
    for di in data:
        di.index(good)

def collect_xovers(xover_glob, bin_size=100e3, DEM=None, min_h=1500, 
                        min_r = 250e3,max_r=2.5e6,
                        asc_minus_desc=None, max_delta_t=24*3600*10, 
                        get_bins_only=False, get_data=False):
    tile_re=re.compile('E(.*)_N(.*).h5')
        
    pad_0=np.arange(-bin_size/2, bin_size/2*1.25, bin_size/4)
    x_pad, y_pad = np.meshgrid(pad_0, pad_0)
    v, d, m, D0, D1 = [[], [], [], [], []]
    files=[]
    for this_glob in xover_glob.split(' '):
        files += glob.glob(this_glob)
    print(len(files))
    xys=[]
    for file in files:
        try:
            xy=np.c_[[int(xx) for xx in tile_re.search(file).groups()]]
            if DEM is not None:
                zz = DEM.interp(xy[0]+x_pad.ravel(), xy[1]+y_pad.ravel())
                zz[~np.isfinite(zz)]=0
                if np.any(zz < min_h):
                    continue
            if min_r is not None:
                if np.all(np.sqrt((xy[0]+x_pad.ravel())**2+ (xy[1]+y_pad.ravel())**2)<min_r):
                    continue
            if max_r is not None:
                if np.any(np.sqrt((xy[0]+x_pad.ravel())**2+ (xy[1]+y_pad.ravel())**2)>max_r):
                    continue
            xys += [xy]
            if get_bins_only:
                continue
            vv, mm, DD = read_xovers(file)
            filter_xovers(vv, mm, DD, min_h=min_h)

            v += [vv]
            
            if get_data:
                D0 += [DD[0]]
                D1 += [DD[1]]
        except Exception as e:
            print(f"problem reading {file} :"
            print(e)
            pass
    if get_bins_only:
        return xys
    v=pc.data(columns=2).from_list(v)
    
    if get_data:
        D0=pc.data(columns=2).from_list(D0)
        D1=pc.data(columns=2).from_list(D1)
        return v, D0, D1
    else:
        return v

def calc_biases_xy(d, v, h_sat, ind=None, field='h_li', b_est=None, max_iterations=20):
    '''
    Calculate per-rgt and per-spot biases based on a set of crossover data
    Inputs:
        d (pc.data) Differences between the interpolated field values 
        v (pc.data) field values interpolated to the crossover location
        h_sat (numeric) : estimated height of the satellite above the reference surface
        ind (numpy array): (optional) indices of crossovers that should be included in the calculation
        field (string): (optinal) field for which to calculate the height differences
        b_est : (optional) estimated bias for the crossovers in 'ind'
        Note the d and v are assumed to represent crossovers within a single cycle
    Returns:
        dictionary containing fields:
            b_spot (numpy array) : Estimated bias for each spot (first entry corresponds to spot 1, etc)
            est_bias_xy (numpy array) : estimated bias due to xy offsets for each data point 
                                        (includes those in the 'ind' variable)
            sigma_hat (float) : robust misfit between the height differences and the bias model
            sigma_corr (float) : standard devlation between height and the full (rgt and xy) bias model
            sigma_uncorr (float) : rss height difference
            sigma_data (float) : standard deviation of raw, edited data
            x_spot, y_spot (numpy arrays): mean spot locations
            t_range: first and last times included in the solution
            fit_index (iterable) : data points included in the fit (after editing)
    '''
    
    if ind is None:
        v1=v.copy()
        d1=d.copy()
        ind = np.arange(len(d.delta_time), dtype=int)
    else:
        # select the data to process
        v1=v[ind]
        d1=d[ind]
    if b_est is not None:
        setattr(d1, field, getattr(d1, field)-b_est)
        
    row=np.arange(v1.shape[0], dtype=int)
    # The first column calculates the x-bias difference
    G_x=((v1.x_sp[:,1]-v1.x_sp[:,0])/(h_sat-np.mean(v1.h_li, axis=1)))[:, None]
    G_y=((v1.y_sp[:,1]-v1.y_sp[:,0])/(h_sat-np.mean(v1.h_li, axis=1)))[:, None]

    # put all the matrices together into one sparse matrix
    G=np.c_[G_x, G_y]
    #G=G.tocsr()
    #make a data vector of height (h_li) differences
    data=getattr(d1, field).ravel()
    
    r=np.zeros(d1.shape[0])
    good=np.ones(G.shape[0], dtype=bool)
    # initialize book-keeping variables
    last_good=np.zeros_like(good)
    sigma=1.e4
    n_data=d1.shape[0]
    count = 0
    # iterate to remove outliers
    while np.sum(np.logical_xor(good,last_good))>0 and (count < max_iterations):
        count += 1
        ii=np.flatnonzero(good)
        mm=sp.linalg.spsolve(G[ii,:].T.dot(G[ii,:]), G[ii,:].T.dot(data[ii]))
        r=getattr(d1, field).ravel()-G[0:d1.size,:].dot(mm)
        sigma_hat=RDE(r[ii[ii<n_data]])
        last_good=good
        good=np.ones(G.shape[0], dtype=bool)
        good[np.argwhere(np.abs(r)>np.maximum(0.05, 3*sigma_hat))]=False
    
    r_temp=r[ii[ii<n_data]]
    # estimate the variance in the corrected data (including the full corrections in h_li)
    sigma_corr = np.sqrt(np.sum(r_temp**2)/(len(r_temp)-G.shape[1]))
    
    r_uncorr=getattr(d1, field).ravel()
    r_temp=r_uncorr[ii[ii<n_data]]
    sigma_uncorr = np.sqrt(np.sum(r_temp**2)/(len(r_temp)-G.shape[1]))
    sigma_data = np.std(getattr(d1, field)[ii[ii<n_data]])

    # estimate the xy component of the biases
    m_xy = mm.copy()
    m_xy[2:]=0
    bias_est_xy = G[0:d1.size,:].dot(m_xy)
    
    fit_index = ind[ii[ii<n_data]]

    # map the solution to the output variables
    N_spot=np.zeros(6)
    
    b_x = mm[0]
    b_y = mm[1]
    x_spot=[np.median(v1.x_sp[v1.spot==spot]) for spot in range(1,7)]
    y_spot=[np.median(v1.y_sp[v1.spot==spot]) for spot in range(1,7)]
    
        
    return {'b_x':b_x, 
            'b_y':b_y, 
            'bias_est_xy' : bias_est_xy,
            'sigma_hat':sigma_hat, 
            'sigma_corr':sigma_corr, 
            'sigma_uncorr':sigma_uncorr,
            'sigma_data':sigma_data,
            'x_spot':x_spot, 
            'y_spot':y_spot,
            't_range': [np.min(v1.delta_time), np.max(v1.delta_time)], 
            'fit_index': fit_index}

def main():
    parser=argparse.ArgumentParser(description="script to estimate time-varying crossover biases", \
                                       fromfile_prefix_chars="@")
    parser.add_argument('--glob_str','-g', type=str, help="glob string that matches the crossover bin files")
    parser.add_argument('--DEM_file', '-D', type=str, help="DEM geotif for selecting crossovers")
    parser.add_argument('--delta_t','-d',  type=float, default=3,  help="maximum time interval for crossover differences, days")
    parser.add_argument('--r_range','-r', type=float, nargs=2, default=[100, 800], help="range of radii around the pole for which bin centers will be read, in km.")
    parser.add_argument('--min_h', type=float, default=1800, help="minimum DEM elevation for which to read crossover bins")
    parser.add_argument('--out_csv', type=str, required=True, help="output csv filename")
    args=parser.parse_args()

    min_r=args.r_range[0]*1000
    min_h=args.min_h
    max_r=args.r_range[1]*1000
    bin_t_tol = args.delta_t*24*3600
    ctr_t_tol = bin_t_tol/2

    DEM=None
    if args.DEM_file is not None:
        DEM=pc.grid.data().from_geotif(args.DEM_file)

    v = collect_xovers(args.glob_str, DEM=DEM, max_delta_t = 2*bin_t_tol, min_r=min_r, min_h=min_h, max_r=max_r) 
    print(v)

    v.cycle_number=np.round(v.cycle_number).astype(int)
    v.rgt=np.round(v.rgt).astype(int)

    hold_info=read_hold_files()
    if hold_info is not None:
        hold_arr=np.c_[hold_info]
        bad=np.zeros_like(v.rgt[:,0], dtype=bool)
        for col in [0, 1]:
            bad |= np.in1d(v.cycle_number[:,col]+1j*v.rgt[:,col], hold_arr[:,0]+1j*hold_arr[:,1])
            good =~bad
        v.index(good)
    uT, t_bins = pc.unique_by_rows(np.round(np.mean(v.delta_time, axis=1)/ctr_t_tol)*ctr_t_tol, return_dict=True)

    v.assign({field:val for field, val in zip(['x_sp','y_sp'], ATL11.calc_xy_spot(v))})
    d=pc.data().from_dict({field:np.diff(getattr(v, field), axis=1) for field in v.fields})

    h_is = 511e3

    out_spot=[]
    out_xy=[]
    out_spot_residual = []
    count=-1
    bin_ctrs=list(t_bins.keys())
    for count, this_t in enumerate(bin_ctrs):
        ind=[]
        # collect the crossovers from three subsequent bins, 
        # then subset them to get only the crossovers within t_bin/2 of the bin
        # center
        for ii in [-2, -1, 0, 1, 2]:
            if (count+ii > 0) & (count+ii < len(bin_ctrs)):
                ind += [t_bins[bin_ctrs[count+ii]]]
        ind=np.concatenate(ind)

        #print(len(ind))
        ind_sub = ind[np.all(np.abs(v.delta_time[ind,:]-this_t) < bin_t_tol/2, axis=1) &
                 np.all(np.isfinite(v.x_sp[ind,:]), axis=1)]    
        if len(ind_sub) < 100:
            continue
        # final fit: run one iteration with the common elements of the two solutions
        out_xy += [calc_biases_xy(d, v, h_is, ind=ind_sub, max_iterations=20)]
        if np.mod(count, 50)==0:
            print(f'{count} out of {len(t_bins)}')

    x_biases = np.c_[[oi['b_x'] for oi in out_xy]]
    y_biases = np.c_[[oi['b_y'] for oi in out_xy]]
    xy_times = np.c_[[np.mean(oi['t_range']) for oi in out_xy]]
    xy_R2 = np.c_[[(oi['sigma_corr']/oi['sigma_uncorr'])**2 for oi in out_xy]]
    x_spot = np.c_[[oi['x_spot'] for oi in out_xy]]
    y_spot = np.c_[[oi['y_spot'] for oi in out_xy]]

    xy_sigma_u = np.c_[[oi['sigma_uncorr'] for oi in out_xy]]
    xy_sigma_c = np.c_[[oi['sigma_corr'] for oi in out_xy]]
    xy_sigma_d = np.c_[[oi['sigma_data'] for oi in out_xy]]

    temp={'delta_time': xy_times}
    temp.update({'x_bias':x_biases,'y_bias':y_biases})


    temp.update({'total_sigma':xy_sigma_d,
                'rgt_corr_sigma':xy_sigma_u,            
                'xy_corr_sigma':xy_sigma_c})
    for field in temp:
        temp[field]=temp[field].ravel()
    df=pd.DataFrame(temp)
    df.to_csv(args.out_csv, index=False)


if __name__=="__main__":
    main()
