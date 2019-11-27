#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:25:40 2019

@author: ben
"""

import glob
import matplotlib.pyplot as plt
import ATL11
#from PointDatabase import mapData, point_data
import pointCollection as pc
import numpy as np


cycles=[3, 4]
cycle_ind=[0,1]

field_dict={'corrected_h':['h_corr','latitude','longitude','delta_time']}

if True:   
    MOG=pc.grid.data().from_geotif('/Volumes/ice1/ben/MOG/2005/mog_2005_1km.tif')
    thedir='/Volumes/ice2/ben/scf/GL_11/U01'
    files=glob.glob(thedir+'/ATL11*.h5')
    xydh=[]
    for count, file in enumerate(files):
        for pair in [1, 2, 3]:
            filePair=(file, pair)    
            D11 = ATL11.data().from_file(filePair[0], pair=filePair[1], field_dict=field_dict).get_xy(None, EPSG=3413) 
            try:
                ind=np.arange(5, D11.x.size, 10)
                temp=pc.data().from_dict({'x':D11.x[ind],'y':D11.y[ind], \
                            'dh':D11.corrected_h.h_corr[ind, cycle_ind[1]]-D11.corrected_h.h_corr[ind, cycle_ind[0]],\
                            'file_ind': np.zeros_like(ind, dtype=int)+count})
                els=(temp.x > MOG.x[0]) & (temp.x < MOG.x[-1]) & \
                    (temp.y > MOG.y[0]) & (temp.y < MOG.y[-1])
                xydh.append(temp[els])
            except:
                print("problem with "+file)
    xydh=pc.data().from_list(xydh)      
     

if True:
    plt.figure()
    MOG.show(cmap='gray', vmin=14000, vmax=17000)
    hl=plt.scatter(xydh.x, xydh.y, 3, linewidth=0, \
                           c=xydh.dh, \
                           vmin=-1.25, vmax=1.25, cmap='Spectral')
    hb=plt.colorbar(shrink=0.5, extend='both')
    hb.set_label(f'cycle {cycles[1]+1} minus cycle {cycles[0]+1}, m')
    
if False:
    plt.figure(1); xy0=plt.ginput()[0]; 
    ATL11_file=files[xydh.file_ind[np.argmin(np.abs(xydh.x+1j*xydh.y - (xy0[0]+1j*xy0[1])))].astype(int)]
    plt.figure(); 
    ATL11_multi_plot(ATL11_file, hemisphere=1)
    
    