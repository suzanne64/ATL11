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
import h5py
import time

cycles=[3, 4, 5, 6]
n_skip=4
field_dict={'corrected_h':['h_corr','h_corr_sigma', 'latitude','longitude','delta_time'], 'ref_surf':['quality_summary', 'dem_h']}
t0=time.time()
if True:
    MOG=pc.grid.data().from_geotif('/Volumes/ice1/ben/MOG/2005/mog_2005_1km.tif')
    thedir='/Volumes/ice2/ben/scf/GL_11/U06'
    files=glob.glob(thedir+'/ATL11*.h5')
    xydh=[]
    data_count=0
    xydh=[None]* (3*len(files))
    for count, file in enumerate(files):
        if np.mod(count, 50)==0:
            print(f'{count} out of {len(files)}, dt={time.time()-t0}')
            t0=time.time()
        for pair in [1, 2, 3]:
            filePair=(file, pair)
            try:
                with h5py.File(file,'r') as h5f:
                    cycle_number=np.array(h5f[f'pt{pair}']['corrected_h']['cycle_number'])

                cycle_ind=[np.flatnonzero(cycle_number==cycles[0])[0], np.flatnonzero(cycle_number==cycles[1])[0]]
                n_cols=len(cycle_number)

                D11 = pc.data().from_h5(file, field_dict={f'/pt{pair}/'+group:field_dict[group] for group in field_dict})

            except Exception:
                continue

            n_data=len(D11.longitude)
            temp=pc.data().from_dict({'latitude':D11.latitude,'longitude':D11.longitude}).get_xy(EPSG=3413)
            D11.assign({'x':temp.x,'y':temp.y})
            # replicated the single-column fields
            for field in ['latitude','longitude','quality_summary','dem_h', 'x','y']:
                setattr(D11, field, np.tile(np.reshape(getattr(D11,field), [n_data, 1]), (1, n_cols)))

            try:
                ind=np.arange(5, D11.x.shape[0], n_skip)
                els=(temp.x[ind] > MOG.x[0]) & (temp.x[ind] < MOG.x[-1]) & \
                    (temp.y[ind] > MOG.y[0]) & (temp.y[ind] < MOG.y[-1])
                ind=ind[els]
                D11.assign({'file_ind':np.zeros_like(D11.x)+count})
                D11.index(ind)
                xydh[data_count]=D11
                data_count += 1

            except:
                print("problem with "+file)

    D=pc.data(columns=len(cycles)).from_list(xydh)


D.to_h5('/Volumes/ice2/ben/scf/GL_11/relU06_dump_every_4th.h5')


if False:
    fig=plt.figure(); plt.clf()
    for col in np.arange(3):
        fig.add_subplot(1,3,col+1)
        MOG.show(cmap='gray', vmin=14000, vmax=17000)
        hl=plt.scatter(D.x[:, col], D.y[:,col], 2, marker='.', c=D.h_corr[:,col+1]-D.h_corr[:,col],
                           vmin=-0.5, vmax=0.5, cmap='Spectral')
        hb=plt.colorbar(shrink=0.75, extend='both')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        hb.set_label(f'cycle {cycles[1]} minus cycle {cycles[0]}, m')
    
if False:
    plt.figure(1); xy0=plt.ginput()[0]; 
    ATL11_file=files[xydh.file_ind[np.argmin(np.abs(xydh.x+1j*xydh.y - (xy0[0]+1j*xy0[1])))].astype(int)]
    plt.figure(); 
    ATL11_multi_plot(ATL11_file, hemisphere=1)
    
    