#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:45:47 2020

@author: ben
"""
import ATL11
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pointCollection as pc

def ATL11_test_plot(ATL11_file, hemisphere=1, pair=2, mosaic=None):
    print(ATL11_file)
    ATL11_rgt=ATL11_file.split('_')[1][:4]
    #
    D = ATL11.data().from_file(ATL11_file, field_dict=None)
    
    if hemisphere==1:
        D.get_xy(EPSG=3413)
    else:
        D.get_xy(EPSG=3031)
      
    cm = matplotlib.cm.get_cmap('jet')
    #colorslist = ['blue','red','orange','purple','brown','pink','gray','olive','cyan','black','yellow']
    colorslist = ['red','darkorange','gold','green','darkturquoise','blue','purple','black','magenta']
    ref, xo, delta = D.get_xovers()

    fig = plt.figure(1)
    if mosaic is not None:
        pc.grid.data().from_geotif(mosaic).show()
        
    crange=[np.nanmin(xo.delta_time), np.nanmax(xo.delta_time)]
    hl = plt.scatter(D.x, D.y, c=D.corrected_h.delta_time[:,0], s=35, marker='.', cmap=cm, vmin=crange[0], vmax=crange[1])
    hl = plt.scatter(ref.x, ref.y, c=xo.delta_time, s=35, marker='x', cmap=cm, vmin=crange[0], vmax=crange[1])
    plt.title('Delta times of ATL11 data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.colorbar(hl)
    
    
    fig = plt.figure(2)
    for ii in range(len(D.corrected_h.cycle_number)):
        cycle=int(D.corrected_h.cycle_number[ii])
        plt.errorbar(D.ref_surf.x_atc, D.corrected_h.h_corr[:,ii], \
                D.corrected_h.h_corr_sigma[:,ii] ,fmt='.',capsize=4,color=colorslist[cycle], label='at for cycle '+str(cycle))
    for cycle in D.corrected_h.cycle_number.astype(int):
        ii=xo.cycle_number==cycle
        im = plt.errorbar(xo.x_atc[ii],xo.h_corr[ii],xo.h_corr_sigma[ii],fmt='x',capsize=4,color='k')
        ii=ref.cycle_number==cycle
        im = plt.errorbar(ref.x_atc[ii],ref.h_corr[ii],ref.h_corr_sigma[ii],fmt='*',capsize=4,color=colorslist[cycle], label='ref, cycle '+str(cycle))

    plt.title('Corrected Heights')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    plt.legend()
    
    fig = plt.figure(3)
    for col in np.arange(1, len(D.corrected_h.cycle_number), dtype=int):
        plt.errorbar(D.ref_surf.x_atc, 
            D.corrected_h.h_corr[:,col]-D.corrected_h.h_corr[:,col-1], \
            np.sqrt(np.sum(D.corrected_h.h_corr_sigma[:, col-1:col+1]**2, \
                           axis=1)) ,fmt='.',capsize=4,color=colorslist[col], 
                label='cycle %d - cycle %d' % (D.corrected_h.cycle_number[col], D.corrected_h.cycle_number[col-1]))
    
    plt.title('Difference in Corrected Heights btn sequential cycles: later minus earlier')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    plt.legend()
    plt.grid()
    
    fig = plt.figure(4)
    ii=np.flatnonzero((ref.cycle_number==4) & (xo.cycle_number==3))
    im = plt.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[0],marker='.') 
    plt.title('Diff Corrected Heights: cyc3-xo(b), cyc4-xo(g)')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    plt.grid()

    fig = plt.figure(5)
    plt.plot(D.corrected_h.h_corr[:,1],'.')
    print('After viewing figures, type Control-C and put cursor over figures, to continue')
    
    fig = plt.figure(5)
    good = np.flatnonzero(np.abs(D.corrected_h.h_corr[:,1])<10000)
    plt.plot(D.corrected_h.h_corr[good,1],'.')
    
    fig = plt.figure(6)
    xo_rgts = np.unique(D.crossing_track_data.rgt).astype('int')
    
    ax1 = fig.add_subplot(211)
    for jj, xo_rgt in enumerate(xo_rgts):
        ii=np.flatnonzero((ref.cycle_number==3) & (xo.cycle_number==3) & (xo.rgt==xo_rgt) & (xo.atl06_quality_summary==0))
        im = ax1.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[jj],marker='.',label=xo_rgt) 
        ii=np.flatnonzero((ref.cycle_number==3) & (xo.cycle_number==3) & (xo.rgt==xo_rgt) & (xo.atl06_quality_summary==1))
        im = ax1.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[jj],marker='x') 
    plt.grid()
    plt.title('Diff Corrected Heights, cycle3:  {0}-xo'.format(ATL11_rgt))
    plt.legend(prop={'size':6})  

    ax2 = fig.add_subplot(212, sharex=ax1)
    for jj, xo_rgt in enumerate(xo_rgts):
        ii=np.flatnonzero((ref.cycle_number==4) & (xo.cycle_number==4) & (xo.rgt==xo_rgt) & (xo.atl06_quality_summary==0))
        im = ax2.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[jj],marker='.',label=xo_rgt) 
        ii=np.flatnonzero((ref.cycle_number==4) & (xo.cycle_number==4) & (xo.rgt==xo_rgt) & (xo.atl06_quality_summary==1))
        im = ax2.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[jj],marker='.') 
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel('Along Track Distance [m]')
    plt.grid()
    plt.title('Diff Corrected Heights, cycle4:  {0}-xo'.format(ATL11_rgt))
    plt.legend(prop={'size':6})  
    
    
    plt.show()
    
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('ATL11_file', type=str)
    parser.add_argument('--Hemisphere','-H', type=int, default=1)
    parser.add_argument('--pair','-p', type=int, default=2)
    parser.add_argument('--mosaic', '-m', type=str)
    args=parser.parse_args()
    ATL11_test_plot(args.ATL11_file, pair=args.pair, hemisphere=args.Hemisphere, mosaic=args.mosaic)


# a good example:
#  -m /Volumes/ice1/ben/MOG/MOG_500.tif -p 3 /Volumes/ice2/ben/scf/GL_11/U03/ATL11_059703_0305_02_vU03.h5