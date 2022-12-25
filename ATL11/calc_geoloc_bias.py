#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:37:43 2022

@author: ben
"""
import numpy as np
import csv

def calc_xy_spot(D):

    H_IS = 511.e3  # Appropriate value of IS2 height WRT WGS84 for Antarctica

    rho = (H_IS - D.h_li)*np.tan(D.ref_coelv * np.pi/180)

    x_spot = -rho * np.cos(-np.pi/180*(D.ref_azimuth-D.seg_azimuth))
    y_spot = -rho * np.sin(-np.pi/180*(D.ref_azimuth-D.seg_azimuth))
    
    return x_spot, y_spot

def calc_geoloc_bias(D, atc_shift_csv_file, atc_shift_table):
    
        
    H_IS=511.e3  # Appropriate value of IS2 height WRT WGS84 for Antarctica

    if atc_shift_csv_file is None:
        D.assign({'dh_geoloc' : np.zeros_like(D.h_li)+np.NaN})
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
    
    D.x_atc -= d_atc['x_bias']
    D.y_atc -= d_atc['y_bias']
        
    return atc_shift_table


