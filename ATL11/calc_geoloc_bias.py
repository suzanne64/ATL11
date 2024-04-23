#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:37:43 2022

@author: ben
"""
import numpy as np
import csv

def calc_xy_spot(D):
    """
    Calculate the along-track and across-track position of spots WRT nadir.

    Parameters
    ----------
    D : pointCollection.data
        ATL06 data structure, containing ref_azimuth, seg_azimuth, and
        ref_coelv fields.

    Returns
    -------
    x_spot : numpy array
        Along-track spot offset WRT nadir.
    y_spot : numpy array
        Across-track spot offset WRT nadir.

    """

    H_IS = 511.e3  # Appropriate value of IS2 height WRT WGS84 for Antarctica

    rho = (H_IS - D.h_li)*np.tan(D.ref_coelv * np.pi/180)

    x_spot = -rho * np.cos(-np.pi/180*(D.ref_azimuth-D.seg_azimuth))
    y_spot = -rho * np.sin(-np.pi/180*(D.ref_azimuth-D.seg_azimuth))

    return x_spot, y_spot

def calc_geoloc_bias(D, atc_shift_csv_file=None, atc_shift_table=None,
                     move_points=False, EPSG=None):
    """
    Calculate and apply vertical bias corrections based on estimated xy errors.

    Parameters
    ----------
    D : pointCollection.data
        ATL06 data structure
    atc_shift_csv_file : str, optional
        Filename for a csv file specifying the estimated x and y shifts to
        apply to the data to correct for geolocation errors. If atc_shift_table
        is specified, the file will not be read. The default is None.
    atc_shift_table : dict, optional
        Table of x and y shifts as a function of delta_time. If this parameter
        is passed, the values in it will be used instead of rereading the
        csv file.  The default is None.

    Returns
    -------
    atc_shift_table : dict
        Table of x and y shifts as a function of delta_time.

    """

    H_IS=511.e3  # Appropriate value of IS2 height WRT WGS84 for Antarctica

    if atc_shift_csv_file is None and atc_shift_table is None:
        D.assign({'dh_geoloc' : np.zeros_like(D.h_li)})
        return

    if atc_shift_table is None:
        atc_shift_table={'delta_time':[],'x_bias':[], 'y_bias':[]}
        with open(atc_shift_csv_file,'r') as fh:
            dr= csv.DictReader(fh)
            for row in dr:
                for field, val in atc_shift_table.items():
                    val += [float(row[field])]
        for key, val in atc_shift_table.items():
            atc_shift_table[key]=np.array(val)

    x_spot, y_spot = calc_xy_spot(D)

    d_atc={}
    for field in ['x_bias','y_bias']:
        d_atc[field] = np.interp(D.delta_time,
                atc_shift_table['delta_time'], atc_shift_table[field])

    # height difference from the satellite
    H0 = H_IS - D.h_li
    # distance to the satellite
    R2 = H0**2 + x_spot**2 + y_spot**2

    # shifted spot location in ATC
    x_spot_corr = x_spot - d_atc['x_bias']
    y_spot_corr = y_spot - d_atc['y_bias']

    # height, corrected
    h_li_c = H_IS - np.sqrt(R2 - (x_spot_corr**2 + y_spot_corr**2))
    D.assign({'dh_geoloc' :  h_li_c - D.h_li })
    if 'x_atc' in D.fields:
        D.x_atc -= d_atc['x_bias']
        D.y_atc -= d_atc['y_bias']

    D.h_li += D.dh_geoloc

    if move_points:
        import pyproj
        #if EPSG is None:
        #    if np.mean(D.latitude) > 0:
        #        EPSG=3413
        #    else:
        #        EPSG=3031
        lonlat=np.array(pyproj.proj.Proj(EPSG)(D.x, D.y, inverse=True)).T
        lonlat[:,1] += 1.e-3
        xy1=np.array(pyproj.proj.Proj(EPSG)(lonlat[:,0], lonlat[:,1])).T
        N_hat = (xy1[:,0]-D.x) + 1j*(xy1[:,1]-D.y)
        N_hat /= np.abs(N_hat)
        x_hat = np.exp(-1j*D.seg_azimuth)*N_hat
        y_hat = 1j*x_hat
        D.x -= (d_atc['x_bias']*np.real(x_hat) + d_atc['y_bias'] * np.real(y_hat))
        D.y -= (d_atc['x_bias']*np.imag(x_hat) + d_atc['y_bias'] * np.imag(y_hat))
    return atc_shift_table
