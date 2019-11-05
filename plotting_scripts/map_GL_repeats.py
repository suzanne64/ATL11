#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:25:40 2019

@author: ben
"""

import glob
import matplotlib.pyplot as plt
import ATL11
from PointDatabase import mapData, point_data
import numpy as np

cycles=[2,3]

field_dict={'corrected_h':['h_corr','latitude','longitude','delta_time']}

if True:
    thedir='/Volumes/ice2/ben/scf/GL_11'
    files=glob.glob(thedir+'/ATL11*.h5')
    D11={}
    for file in files:
        for pair in [1, 2, 3]:
            filePair=(file, pair)    
            D11[filePair] = ATL11.data().from_file(filePair[0], pair=filePair[1]).get_xy(None, EPSG=3413) 
    
    MOG=mapData().from_geotif('/Volumes/ice1/ben/MOG/2005/mog_2005_1km.tif')
    
if True:
    plt.figure()
    MOG.show(cmap='gray', vmin=14000, vmax=17000)
    for ii, ff in enumerate(D11.keys()):
        els=(D11[ff].x > MOG.x[0]) & (D11[ff].x < MOG.x[-1]) & \
        (D11[ff].y > MOG.y[0]) & (D11[ff].y < MOG.y[-1])
        
        if not np.any(els):
            continue
        try:
            hl=plt.scatter(D11[ff].x[els][::5], D11[ff].y[els][::5], 3, linewidth=0, \
                           c=(D11[ff].corrected_h.h_corr[els,cycles[1]]-D11[ff].corrected_h.h_corr[els,cycles[0]])[::5], \
                           vmin=-1.5, vmax=1.5, cmap='Spectral')
        except:
            print(ff)
            pass
    xyf=[]
    for count, ff in enumerate(D11.keys()):
        xyf.append(point_data().from_dict({'x':D11[ff].x, 'y':D11[ff].y, 'N': count+np.zeros_like(D11[ff].x)}))
    xyf=point_data().from_list(xyf)
    xyf.index(np.isfinite(xyf.x))
    
    hb=plt.colorbar(shrink=0.5, extend='both')
    hb.set_label(f'cycle {cycles[1]+1} minus cycle {cycles[0]+1}, m')
    
if False:
    plt.figure(1); xy0=plt.ginput()[0]; ATL11_file=list(D11.keys())[xyf.N[np.argmin(np.abs(xyf.x+1j*xyf.y - (xy0[0]+1j*xy0[1])))].astype(int)][0]
    plt.figure(); ATL11_multi_plot(ATL11_file)
    
    