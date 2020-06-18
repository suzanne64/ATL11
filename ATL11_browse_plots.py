#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:45:47 2020

@author: ben05
"""
import ATL11
import numpy as np
from scipy import stats
import sys, os, h5py, glob
import io
import pointCollection as pc
from PointDatabase.mapData import mapData

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import osgeo.gdal
import imageio
#import wradlib as wrl


def ATL11_browse_plots(ATL11_file, hemisphere=1, mosaic=None, out_path=None, pdf=False):
    print('File to plot',os.path.basename(ATL11_file))
    ATL11_file_str = os.path.basename(ATL11_file).split('.')[0]
    if out_path is None:
        out_path = os.path.dirname(ATL11_file)
    

    cm = mpl.cm.get_cmap('magma')
    colorslist = ['black','darkred','red','darkorange','gold','yellowgreen','green','darkturquoise','steelblue','blue','purple','orchid','deeppink']
    cmpr = ['red','green','blue']
    sec2year = 60*60*24*365.25
    ipair   = [0]
    ipairxo = [0]

    # gather all pairs of data
    for pr in np.arange(3): 
        pair = pr+1

        D = ATL11.data().from_file(ATL11_file, pair=pair, field_dict=None)
        if pair == 1:
            start_cycle=D.corrected_h.cycle_number[0]
            end_cycle=D.corrected_h.cycle_number[-1]
            num_cycles=len(D.corrected_h.cycle_number)
            cmCount = ListedColormap(colorslist[0:num_cycles+1])
            cmCycles = ListedColormap(colorslist[np.int(start_cycle):np.int(end_cycle)+1])

        if hemisphere==1:
            D.get_xy(EPSG=3413)
            # set cartopy projection as EPSG 3413
            projection = ccrs.Stereographic(central_longitude=-45.0, 
                                            central_latitude=+90.0,
                                            true_scale_latitude=+70.0)   
        else:
            D.get_xy(EPSG=3031)
            projection = ccrs.Stereographic(central_longitude=+0.0,
                                            central_latitude=-90.0,
                                            true_scale_latitude=-71.0)  
        
        if pair == 1:
            h_corr     = D.corrected_h.h_corr
            delta_time = D.corrected_h.delta_time
            ref_pt     = D.corrected_h.ref_pt
            lat        = D.corrected_h.latitude
            lon        = D.corrected_h.longitude
            x = D.x
            y = D.y
            dem_h      = D.ref_surf.dem_h

            ref, xo, delta   = D.get_xovers()  
            ref_h_corr       = ref.h_corr
            ref_cycle_number = ref.cycle_number
            xo_h_corr        = xo.h_corr
            delta_h_corr     = delta.h_corr
            xo_ref_pt        = xo.ref_pt
            xo_cycle_number  = xo.cycle_number
            xo_atl06_quality_summary = xo.atl06_quality_summary
            ipairxo.append(ipairxo[-1]+xo.h_corr.shape[0])                
                
        else:
            h_corr     = np.concatenate( (h_corr,D.corrected_h.h_corr), axis=0) 
            delta_time = np.concatenate( (delta_time,D.corrected_h.delta_time), axis=0) 
            ref_pt     = np.concatenate( (ref_pt,D.corrected_h.ref_pt), axis=0) 
            lat        = np.concatenate( (lat,D.corrected_h.latitude), axis=0) 
            lon        = np.concatenate( (lon,D.corrected_h.longitude), axis=0) 
            x = np.concatenate( (x,D.x), axis=0)
            y = np.concatenate( (y,D.y), axis=0)
            dem_h = np.concatenate( (dem_h,D.ref_surf.dem_h), axis=0)

            ref, xo, delta   = D.get_xovers()
            ref_h_corr       = np.concatenate( (ref_h_corr, ref.h_corr), axis=0)
            ref_cycle_number = np.concatenate( (ref_cycle_number, ref.cycle_number), axis=0)
            xo_h_corr        = np.concatenate( (xo_h_corr, xo.h_corr), axis=0)
            delta_h_corr     = np.concatenate( (delta_h_corr, delta.h_corr), axis=0)
            xo_ref_pt        = np.concatenate( (xo_ref_pt, xo.ref_pt), axis=0)
            xo_cycle_number  = np.concatenate( (xo_cycle_number, xo.cycle_number), axis=0)
            xo_atl06_quality_summary = np.concatenate( (xo_atl06_quality_summary, xo.atl06_quality_summary), axis=0)
            ipairxo.append(ipairxo[-1]+xo.h_corr.shape[0])                

        ipair.append(ipair[-1]+D.corrected_h.h_corr.shape[0])
        
    xctr = (np.max(x) - np.min(x))/2 + np.min(x)
    yctr = (np.max(y) - np.min(y))/2 + np.min(y)
    xwidth = np.max(x) - np.min(x)
    ywidth = np.max(y) - np.min(y)
    if ywidth > xwidth:
        xbuf = ywidth/4/2;
        ybuf = ywidth/2 + 1e4;
    else:
        xbuf = xwidth/2 + 1e4;
        ybuf = xwidth/4/2;
    bounds = [ [np.min([xctr-xbuf, np.min(x)-1e4]), np.max([xctr+xbuf, np.max(x)+1e4])],
               [np.min([yctr-ybuf, np.min(y)-1e4]), np.max([yctr+ybuf, np.max(y)+1e4])] ]
    

    ddem = h_corr-dem_h[:,None]

    # get dHdt for whatever cycles have some valid points, prefereably over the longest time range.
    ccl = -1   # last cycle index
    ccf = 0    # first cycle index
    dHdt = np.full([len(h_corr[:,0]),], np.nan)
    while np.all(np.isnan(h_corr[:,ccl])):
        ccl -= 1
    else:
        while np.all(np.isnan(h_corr[:,ccf])):
            ccf += 1
        else:
            if len(h_corr[0,ccf:ccl]) == 0:
                pass
            else:
                dHdt = ( (h_corr[:,ccl] - h_corr[:,ccf]) / (delta_time[:,ccl] - delta_time[:,ccf]) ) * sec2year

    # get min, max values for y-axis
    if np.any(~np.isnan(dHdt)):
        dHdt05 = stats.scoreatpercentile(dHdt[~np.isnan(dHdt)].ravel(),5)
        dHdt95 = stats.scoreatpercentile(dHdt[~np.isnan(dHdt)].ravel(),95)
    else:
        dHdt05 = np.nan
        dHdt95 = np.nan
    h05 = stats.scoreatpercentile(h_corr[~np.isnan(h_corr)].ravel(),5)
    h95 = stats.scoreatpercentile(h_corr[~np.isnan(h_corr)].ravel(),95)
    ddem05 = stats.scoreatpercentile(ddem[~np.isnan(ddem)].ravel(),5)
    ddem95 = stats.scoreatpercentile(ddem[~np.isnan(ddem)].ravel(),95)

    try:
        if isinstance(xo_h_corr, np.ndarray): 
            goodxo = np.logical_and(~np.isnan(ref_h_corr), ~np.isnan(xo_h_corr))
            dxo05 = stats.scoreatpercentile(ref_h_corr[goodxo]-xo_h_corr[goodxo],5)
            dxo95 = stats.scoreatpercentile(ref_h_corr[goodxo]-xo_h_corr[goodxo],95)
    except Exception as E:
        pass 
    
    if mosaic is not None:  # used for Figure 1
        DEM = pc.grid.data().from_geotif(mosaic, bounds=bounds)        
        DEM.z = np.gradient(DEM.z)[0]
        gz05 = stats.scoreatpercentile(DEM.z[~np.isnan(DEM.z)], 5)
        gz95 = stats.scoreatpercentile(DEM.z[~np.isnan(DEM.z)], 95)                

#    with h5py.File('goddammit.h5','w') as hftest:
    figs = []
    # make plots,         
    if len(DEM.y) >= len(DEM.x):    
        fig1, ax1 = plt.subplots(1,3,sharex=True,sharey=True) #, subplot_kw=dict(projection=projection))
    else:
        fig1, ax1 = plt.subplots(3,1,sharex=True,sharey=True) #, subplot_kw=dict(projection=projection))
    if mosaic is not None:
        for ii in np.arange(3):
            DEM.show(ax=ax1[ii], xy_scale=1/1000, cmap='gray', \
                           vmin=gz05, vmax=gz95, interpolation='nearest', aspect='equal')
    h0 = ax1[0].scatter(x/1000, y/1000, c=h_corr[:,ccl]/1000, s=2, cmap=cm, marker='.', vmin=h05/1000, vmax=h95/1000)  #h05/1000, vmax=h95/1000#norm=normh_corr, 
    ax1[0].set_title('Heights, Cycle {}, km'.format(np.int(D.corrected_h.cycle_number[ccl])), fontdict={'fontsize':10});
    h1 = ax1[1].scatter(x/1000, y/1000, c=np.count_nonzero(~np.isnan(h_corr),axis=1), s=2, marker='.', cmap=cmCount, vmin=0-0.5, vmax=num_cycles+0.5)
    if np.any(~np.isnan(dHdt)):
        h2 = ax1[2].scatter(x/1000, y/1000, c=dHdt, s=2, marker='.', cmap=cm, vmin=dHdt05, vmax=dHdt95)
        ax1[2].set_title('dH/dt, m/yr', fontdict={'fontsize':10});
        if hemisphere==1:
            plt.figtext(0.1,0.01,'Figure 1. Height data, in km, from cycle {0} (1st panel). Number of cycles with valid height data (2nd panel). Change in height over time, in meters/year, cycle {0} from cycle {1} (3rd panel). All overlaid on gradient of DEM. x, y in km. Maps are plotted in a polar-stereographic projection with a central longitude of 45E and a standard latitude of 70N.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
        elif hemisphere==-1:
            plt.figtext(0.1,0.01,'Figure 1. Height data, in km, from cycle {0} (1st panel). Number of cycles with valid height data (2nd panel). Change in height over time, in meters/year, cycle {0} from cycle {1} (3rd panel). All overlaid on gradient of DEM. x, y in km. Maps are plotted in a polar-stereographic projection with a central longitude of 0E and a standard latitude of 71S.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
    else:
        h2 = ax1[2].scatter(x/1000, y/1000, c=h_corr[:,ccf]/1000, s=2, cmap=cm, marker='.', vmin=h05/1000, vmax=h95/1000)  #norm=normh_corr, 
        ax1[2].set_title('Heights, Cycle {}, km'.format(np.int(D.corrected_h.cycle_number[ccf])), fontdict={'fontsize':10});
        if hemisphere==1:
            plt.figtext(0.1,0.01,'Figure 1. Height data, in km, from cycle {0} (1st panel). Number of cycles with valid height data (2nd panel). Height data, in km, from cycle {1} (3rd panel). All overlaid on gradient of DEM. x, y in km. Maps are plotted in a polar-stereographic projection with a central longitude of 45E and a standard latitude of 70N.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
        elif hemisphere==-1:
            plt.figtext(0.1,0.01,'Figure 1. Height data, in km, from cycle {0} (1st panel). Number of cycles with valid height data (2nd panel). Height data, in km, from cycle {1} (3rd panel). All overlaid on gradient of DEM. x, y in km. Maps are plotted in a polar-stereographic projection with a central longitude of 0E and a standard latitude of 71S.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
        
    ax1[0].set_ylabel('y [km]', fontdict={'fontsize':10})
    ax1[1].set_title('Number of Valid Heights', fontdict={'fontsize':10});
    fig1.colorbar(h0, ax=ax1[0]) 
    fig1.colorbar(h1, ticks=np.arange(num_cycles+1), ax=ax1[1]) 
    fig1.colorbar(h2, ax=ax1[2]) 
    fig1.suptitle('{}'.format(os.path.basename(ATL11_file)))
    plt.subplots_adjust(bottom=0.23, top=0.9)
    fig1.savefig('{0}/{1}_Figure1_h_corr_NumValids_dHdtOverDEM.png'.format(out_path,ATL11_file_str),format='png', bbox_inches='tight')
    fig1.savefig('{0}/{1}_BRW_default1.png'.format(out_path,ATL11_file_str),format='png')

#            plt2hdf5(hftest, fig1, 'images/default0')
#            imgdata = io.BytesIO()
#            fig1.savefig(imgdata, format='png')
#            imgdata.seek(0)
#            img = imageio.imread(imgdata, pilmode='RGB')
#            with h5py.File('goddammit.h5','w') as hftest:
#                dset = hftest.create_dataset('imagesc/default0', img.shape, data=img.data,\
#                                             chunks=img.shape, compression='gzip', compression_opts=6)
#                dset.attrs['CLASS'] = np.string_('IMAGE')
#                dset.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
#                dset.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')
    #exit(-1)
    
    fig2,ax2 = plt.subplots()
    hist, bin_edges = np.histogram(np.count_nonzero(~np.isnan(h_corr),axis=1), bins=np.arange((num_cycles)+2))
    valid_dict = {}
    for kk in np.arange(np.max(bin_edges)):
        valid_dict.update( {bin_edges[kk]: colorslist[kk]} )
    ax2.bar(bin_edges[:-1],hist,color=[valid_dict[r] for r in np.arange(np.max(bin_edges))])
    ax2.set_xticks(bin_edges[:-1])
    fig2.suptitle('{}'.format(os.path.basename(ATL11_file)))
    plt.figtext(0.1,0.01,'Figure 2. Histogram of number of cycles with valid height measurements, all beam pairs.',wrap=True)
    plt.subplots_adjust(bottom=0.15)
    fig2.savefig('{0}/{1}_Figure2_validRepeats_hist.png'.format(out_path,ATL11_file_str),format='png')
  
    if num_cycles <= 3:  
        fig5,ax5 = plt.subplots(num_cycles,1,sharex=True,sharey=True)
    elif num_cycles == 4:
        fig5,ax5 = plt.subplots(2,2,sharex=True,sharey=True)
    elif num_cycles >= 5 and num_cycles <= 6:
        fig5,ax5 = plt.subplots(3,2,sharex=True,sharey=True)
    elif num_cycles >= 7 and num_cycles <= 9:
        fig5,ax5 = plt.subplots(3,3,sharex=True,sharey=True)
    elif num_cycles >= 10:
        fig5,ax5 = plt.subplots(3,4,sharex=True,sharey=True)
    for ii, ax in enumerate(ax5.reshape(-1)):
        if ii<ddem.shape[-1]:
            ax.hist(ddem[:,ii],bins=np.arange(np.floor(ddem05*10)/10,np.ceil(ddem95*10)/10+0.1,0.1), color=colorslist[np.int(D.corrected_h.cycle_number[ii])])  
            if ii == 0:
                ax.set_title('height-DEM: Cycle {}'.format(np.int(D.corrected_h.cycle_number[ii])), fontdict={'fontsize':10})
            else:
                ax.set_title('Cycle {}'.format(np.int(D.corrected_h.cycle_number[ii])), fontdict={'fontsize':10})
    plt.figtext(0.1,0.01,'Figure 5. Histogram of corrected_h/h_corr heights minus DEM, in meters. One historgram per cycle, all beam pairs. X-axis limits are the scores at 5% and 95%.',wrap=True)
    plt.subplots_adjust(bottom=0.15)
    fig5.suptitle('{}'.format(os.path.basename(ATL11_file)))
    fig5.savefig('{0}/{1}_Figure5_h_corr-DEM_hist.png'.format(out_path,ATL11_file_str),format='png')

    for pr in np.arange(3):
        pair=pr+1
        
        if pair == 1:
            fig3,ax3 = plt.subplots(1,3,sharex=True,sharey=True)                
            cycle_dict = {}
            for cc in np.arange(start_cycle, end_cycle+1):
                cycle_dict.update( {np.int(cc):colorslist[np.int(cc)]} )
            fig3.subplots_adjust(bottom=0.15)
            plt.figtext(0.1,0.01,'Figure 3. Number of valid height measurements from each beam pair.',wrap=True)

        which_cycles = np.array([])
        for ii, cc in enumerate(np.arange(start_cycle, end_cycle+1)):
            which_cycles = np.concatenate( (which_cycles,(np.int(cc)*np.ones(np.count_nonzero(~np.isnan(h_corr[ipair[pr]:ipair[pr+1]-1,ii]))).ravel())), axis=0)
        hist, bin_edges = np.histogram(which_cycles, bins=np.arange(start_cycle, end_cycle+2))
        ax3[pr].bar(D.corrected_h.cycle_number[:], hist, color=[cycle_dict[np.int(r)] for r in D.corrected_h.cycle_number[:]])
        ax3[pr].set_xlim((D.corrected_h.cycle_number[0]-0.5, D.corrected_h.cycle_number[-1]+0.5))
        ax3[pr].set_xticks(D.corrected_h.cycle_number)
        ax3[pr].set_title('Beam Pair {}'.format(pair))
        if pair == 3:
            #ax3[1].set_title('Number of valid heights from each pair', fontdict={'fontsize':10})
            ax3[1].set_xlabel('cycle number', fontdict={'fontsize':10})
            fig3.suptitle('{}'.format(os.path.basename(ATL11_file)))
            fig3.savefig('{0}/{1}_Figure3_validRepeatsCycle_hist.png'.format(out_path,ATL11_file_str),format='png')
            fig3.savefig('{0}/{1}_BRW_default2.png'.format(out_path,ATL11_file_str),format='png')
                 
        if pair == 1:
            fig4, ax4 = plt.subplots(2,3,sharex=True,sharey='row')
            plt.figtext(0.1,0.01,'Figure 4. Top row: Heights, in meters, plotted for each beam pair: 1 (left), 2 (center), 3 (right). Bottom row: Heights minus DEM, in meters. Y-axis limits are scores at 5% and 95%. Color coded by cycle number. Plotted against reference point number/1000.',wrap=True)
            labels=[]
        for ii, cyc in enumerate(np.arange(start_cycle, end_cycle+1)):
            labels.append('cycle {:d}'.format(np.int(cyc)))
            ax4[0,pr].plot(ref_pt[ipair[pr]:ipair[pr+1]-1]/1000,h_corr[ipair[pr]:ipair[pr+1]-1,ii], '.', markersize=1, color=colorslist[np.int(cyc)], linewidth=0.5)                
            ax4[1,pr].plot(ref_pt[ipair[pr]:ipair[pr+1]-1]/1000,ddem[ipair[pr]:ipair[pr+1]-1,ii], '.', markersize=1, color=colorslist[np.int(cyc)], linewidth=0.5)
        ax4[0,1].set_title('corrected_h/h_corr', fontdict={'fontsize':10});
        ax4[0,pr].grid(linestyle='--')
        ax4[1,1].set_title('corrected_h/h_corr minus DEM', fontdict={'fontsize':10});
        ax4[1,pr].grid(linestyle='--')
        ax4[0,0].set_ylim((h05,h95))
        ax4[0,0].set_ylabel('meters')
        ax4[1,0].set_ylim((ddem05,ddem95))
        ax4[1,0].set_ylabel('meters')
        plt.suptitle('{}'.format(os.path.basename(ATL11_file)))
        if pair == 3:
            fig4.subplots_adjust(bottom=0.2, right=0.8)
            cbar_ax = fig4.add_axes([0.85, 0.2, 0.02, 0.67])
            cmap = plt.get_cmap(cmCycles,num_cycles+1)
            norm = mpl.colors.Normalize(vmin=start_cycle, vmax=end_cycle)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            deltac=(end_cycle-start_cycle)/(num_cycles)
            cbar = fig4.colorbar(sm, ticks=np.arange(start_cycle+deltac/2,end_cycle+deltac,deltac), cax=cbar_ax)
            cbar.set_ticklabels(np.arange(np.int(start_cycle),np.int(end_cycle)+1))
            cbar.set_label('Cycle Number')
            fig4.savefig('{0}/{1}_Figure4_h_corr_h_corr-DEM.png'.format(out_path,ATL11_file_str),format='png')

        if pair == 1:
            fig6, ax6 = plt.subplots()
            plt.figtext(0.1,0.01,'Figure 6. Change in height over time, dH/dt, in meters/year. dH/dt is cycle {0} from cycle {1}. Color coded by beam pair: 1 (red), 2 (green), 3 (blue). Y-axis limits are scores at 5% and 95%. Plotted against reference point number/1000.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
        if np.any(~np.isnan(dHdt[ipair[pr]:ipair[pr+1]-1])):
            ax6.plot(ref_pt[ipair[pr]:ipair[pr+1]-1]/1000,dHdt[ipair[pr]:ipair[pr+1]-1], '.', markersize=1, color=cmpr[pr] )
        if pair == 3:
            if np.any(~np.isnan(dHdt[ipair[pr]:ipair[pr+1]-1])):
                ax6.set_ylim([dHdt05,dHdt95])
            else:
                ax6.set_ylim([-1,1])
            ax6.grid(linestyle='--',linewidth=0.3)
            ax6.set_title('Change in height over time: cycle {0} minus cycle {1}'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])), fontdict={'fontsize':10})
            ax6.set_ylabel('meters/year')
            fig6.suptitle('{}'.format(os.path.basename(ATL11_file)))
            fig6.subplots_adjust(bottom=0.2,right=0.8)
            cbar_ax = fig6.add_axes([0.85, 0.2, 0.02, 0.67])
            cmap = plt.get_cmap(ListedColormap(cmpr))
            norm = mpl.colors.Normalize(vmin=1, vmax=3)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            deltac=(3-1)/(3)
            cbar = fig6.colorbar(sm, ticks=np.arange(1+deltac/2,3+deltac,deltac), cax=cbar_ax)
            cbar.set_ticklabels(np.arange(1,3+1))
            cbar.set_label('Beam Pair')
            fig6.savefig('{0}/{1}_Figure6_dHdt.png'.format(out_path,ATL11_file_str),format='png')

        if pair == 1:
            fig7,ax7 = plt.subplots(1,3,sharex=True,sharey=True)
            fig7.subplots_adjust(bottom=0.2)
            plt.figtext(0.1,0.01,'Figure 7. Histograms of change in height over time, dH/dt, in meters/year. dH/dt is cycle {0} from cycle {1}. One histogram per beam pair: 1 (red), 2 (green), 3 (blue). X-axis limits are the scores at 5% and 95%.'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[ccf])),wrap=True)
        if np.any(~np.isnan(dHdt[ipair[pr]:ipair[pr+1]-1])):
            ax7[pr].hist(dHdt[ipair[pr]:ipair[pr+1]-1], bins=np.arange(np.floor(dHdt05*10)/10,np.ceil(dHdt95*10)/10+0.1,0.1), color=cmpr[pr])
        ax7[pr].grid(linestyle='--')
        ax7[1].set_title('Change in height histograms: cycle {0} minus cycle {1}'.format(np.int(D.corrected_h.cycle_number[ccl]),np.int(D.corrected_h.cycle_number[0])), fontdict={'fontsize':10})
        fig7.suptitle('{}'.format(os.path.basename(ATL11_file)))
        if pair == 3:
            fig7.savefig('{0}/{1}_Figure7_dHdt_hist.png'.format(out_path,ATL11_file_str),format='png')

        if pair==1:
            fig8, ax8 = plt.subplots(2, 3, sharey='row', sharex=True)
            plt.figtext(0.1,0.01,'Figure 8. Top row: Heights from crossing track data, in meters, plotted for each beam pair: 1 (left), 2 (center), 3 (right). Bottom row: Heights minus crossing track heights. Y-axis limits are scores at 5% and 95%. Color coded by cycle number. Plotted against reference point number/1000.',wrap=True)
        if isinstance(xo_h_corr, np.ndarray): 
            for ii, cyc in enumerate(D.corrected_h.cycle_number):
                cc=np.flatnonzero((xo_cycle_number[ipairxo[pr]:ipairxo[pr+1]-1]==cyc) & (xo_atl06_quality_summary[ipairxo[pr]:ipairxo[pr+1]-1]==0))
                ax8[0,pr].plot(xo_ref_pt[ipairxo[pr]:ipairxo[pr+1]-1][cc]/1000,xo_h_corr[ipairxo[pr]:ipairxo[pr+1]-1][cc],'x',color=colorslist[np.int(cyc)], markersize=1, label='cycle {:d}'.format(np.int(cyc)));
                ax8[0,pr].grid(linestyle='--')
                ccc=np.flatnonzero((xo_cycle_number[ipairxo[pr]:ipairxo[pr+1]-1]==cyc) & (ref_cycle_number[ipairxo[pr]:ipairxo[pr+1]-1]==cyc) & (xo_atl06_quality_summary[ipairxo[pr]:ipairxo[pr+1]-1]==0))
                ax8[1,pr].plot(xo_ref_pt[ipairxo[pr]:ipairxo[pr+1]-1][ccc]/1000,ref_h_corr[ipairxo[pr]:ipairxo[pr+1]-1][ccc]-xo_h_corr[ipairxo[pr]:ipairxo[pr+1]-1][ccc], '.', color=colorslist[np.int(cyc)], markersize=1, label=None);
                ax8[1,pr].grid(linestyle='--')
    
            ax8[0,0].set_ylim((h05, h95))
            ax8[0,0].set_ylabel('meters')
            ax8[0,1].set_title('crossing_track_data/h_corr', fontdict={'fontsize':10})
            ax8[1,0].set_ylabel('meters')
            ax8[1,0].set_ylim((dxo05,dxo95))
            ax8[1,1].set_title('corrected_h/h_corr minus crossing_track_data/h_corr', fontdict={'fontsize':10})
        else:
            ax8[0,0].text(0.2,0.5,'No cross over data in this file')
        if pair == 3:
            fig8.subplots_adjust(bottom=0.2,right=0.8)
            cbar_ax = fig8.add_axes([0.85, 0.2, 0.02, 0.67])
            cmap = plt.get_cmap(cmCycles,num_cycles+1)
            norm = mpl.colors.Normalize(vmin=start_cycle, vmax=end_cycle)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            deltac=(end_cycle-start_cycle)/(num_cycles)
            cbar = fig4.colorbar(sm, ticks=np.arange(start_cycle+deltac/2,end_cycle+deltac,deltac), cax=cbar_ax)
            cbar.set_ticklabels(np.arange(np.int(start_cycle),np.int(end_cycle)+1))
            cbar.set_label('Cycle Number')
            plt.suptitle('{}'.format(os.path.basename(ATL11_file)))
            fig8.savefig('{0}/{1}_Figure8_h_corr_h_corr-CrossOver.png'.format(out_path,ATL11_file_str),format='png')

    if pdf:    #save them all to one .pdf file
        figs = list(map(plt.figure, plt.get_fignums()))
        with PdfPages('{0}/{1}.pdf'.format(out_path,ATL11_file_str)) as pdff:
            for fig in figs:
                pdff.savefig(fig)

    # put images into browse file            
    ATL11_file_brw='{}/{}_BRW.h5'.format(out_path,ATL11_file_str)
    if os.path.isfile(ATL11_file_brw):
        os.remove(ATL11_file_brw)
    
    with h5py.File(ATL11_file_brw,'w') as hf:
        for ii, name in enumerate(sorted(glob.glob('{0}/{1}_BRW_def*.png'.format(out_path,ATL11_file_str)))):
            img = imageio.imread(name, pilmode='RGB') 
    
            namestr = os.path.splitext(name)[0]
            namestr = os.path.basename(namestr).split('BRW_')[-1]
            dset = hf.create_dataset('default/'+namestr, img.shape, data=img.data, \
                                     chunks=img.shape, compression='gzip',compression_opts=6)
            dset.attrs['CLASS'] = np.string_('IMAGE')
            dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
            dset.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
            dset.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')
        for ii, name in enumerate(sorted(glob.glob('{0}/{1}_Figure*.png'.format(out_path,ATL11_file_str)))):
            if 'Figure1' not in name and 'Figure3' not in name:
                img = imageio.imread(name, pilmode='RGB') 
        
                namestr = os.path.splitext(name)[0]
                namestr = os.path.basename(namestr).split('Figure')[-1]
                dset = hf.create_dataset(namestr[2:], img.shape, data=img.data, \
                                         chunks=img.shape, compression='gzip',compression_opts=6)
                dset.attrs['CLASS'] = np.string_('IMAGE')
                dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
                dset.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
                dset.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')
        with h5py.File(ATL11_file,'r') as g:
            g.copy('ancillary_data',hf)
        
    for name in sorted(glob.glob('{0}/{1}_Figure*.png'.format(out_path,ATL11_file_str))):
        if os.path.isfile(name): os.remove(name)
    
#    plt.show()
#
    
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('ATL11_file', type=str)
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='1 for Norhtern, -1 for Southern')
    parser.add_argument('--mosaic', '-m', type=str)
    parser.add_argument('--out_path', '-o', type=str, help='default is ATL11_file path')
    parser.add_argument('--pdf', action='store_true', default=False, help='write images to .pdf file')
    args=parser.parse_args()
    ATL11_browse_plots(args.ATL11_file, hemisphere=args.Hemisphere, mosaic=args.mosaic, out_path=args.out_path, pdf=args.pdf)



