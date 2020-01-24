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

def ATL11_test_plot(ATL11_file):
    print(ATL11_file)
    #
    D = ATL11.data().from_file(ATL11_file, field_dict=None)
    cm = matplotlib.cm.get_cmap('jet')
    colorslist = ['blue','red','orange','purple','brown','pink','gray','olive','cyan','black','yellow']
    ref, xo, delta = D.get_xovers()

    fig = plt.figure(1)
    crange=[np.nanmin(xo.delta_time), np.nanmax(xo.delta_time)]
    im = plt.scatter(D.corrected_h.longitude, D.corrected_h.latitude, c=D.corrected_h.delta_time[:,0], s=35, marker='.', cmap=cm, vmin=crange[0], vmax=crange[1])
    im = plt.scatter(D.crossing_track_data.longitude, D.crossing_track_data.latitude, c=D.crossing_track_data.delta_time, s=35, marker='.', cmap=cm, vmin=crange[0], vmax=crange[1])
    plt.title('Delta times of ATL11 data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.colorbar(im)
    
    
    fig = plt.figure(2)
    for ii in [0,1]:
        plt.errorbar(D.ref_surf.x_atc, D.corrected_h.h_corr[:,ii], \
                D.corrected_h.h_corr_sigma[:,ii] ,fmt='.',capsize=4,color=colorslist[ii])
    for cycle in [3, 4]:
        ii=xo.cycle_number==cycle
        #im = plt.errorbar(ref.x_atc[],ref.h_corr,ref.h_corr_sigma,fmt='.',capsize=4,color=colorslist[0]) 
        im = plt.errorbar(xo.x_atc[ii],xo.h_corr[ii],xo.h_corr_sigma[ii],fmt='x',capsize=4,color='k')
        ii=ref.cycle_number==cycle
        im = plt.errorbar(ref.x_atc[ii],ref.h_corr[ii],ref.h_corr_sigma[ii],fmt='*',capsize=4,color='g')

    plt.title('Corrected Heights: cyc3(b), cyc4(g), crossing :x, ref:*')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    
    fig = plt.figure(3)
    plt.errorbar(D.ref_surf.x_atc, D.corrected_h.h_corr[:,1]-D.corrected_h.h_corr[:,0], \
                np.sqrt(np.sum(D.corrected_h.h_corr_sigma**2, axis=1)) ,fmt='.',capsize=4,color=colorslist[0])
    
    plt.title('Difference in Corrected Heights btn sequential cycles: later minus earlier')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    plt.grid()
    
    fig = plt.figure(4)
    ii=np.flatnonzero((ref.cycle_number==4) & (xo.cycle_number==3))
    im = plt.scatter(ref.x_atc[ii], (xo.h_corr[ii]-ref.h_corr[ii].ravel()), c=colorslist[0],marker='.') 
    plt.title('Diff Corrected Heights: cyc3-xo(b), cyc4-xo(g)')
    plt.xlabel('Along Track Distance [m]')
    plt.ylabel('Heights [m]')
    plt.grid()
    print('After viewing figures, type Control-C and put cursor over figures, to continue')
    plt.show()
    
if __name__=='__main__':
    ATL11_test_plot(sys.argv[1])
    