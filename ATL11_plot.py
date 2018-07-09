# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:05:50 2018

@author: ben
"""
import matplotlib.pyplot as plt
import numpy as np

class ATL11_plot:
    def __init__(self, D11):
        self.D11=D11
        
        fig=plt.figure()
        
        self.ax2=plt.axes([0.05,  0.05, 0.4, 0.375])
        self.ax3=plt.axes([0.55,  0.05, 0.4, 0.375])
        self.ax1=plt.axes([0.05, 0.525, 0.9, 0.375])
   
        self.h_errorbars=D11.plot()

        #fig.canvas.mpl_connect('pick_event', self.pick_event)
        fig.canvas.mpl_connect('button_press_event', self.pick_event)

        plt.show(block=True)
        return
        
    def pick_event(self, event):
         
        #xx=event.artist.get_xdata()
        xx=event.xdata
        #ii=event.ind
        D11=self.D11
        x0=D11.ref_surf.ref_pt_x_atc
        this=np.argmin(np.abs(x0-xx))
        #this=np.flatnonzero(x0==xx[ii])
        print("this is %d" % this)
        print "x0=%d" % x0[this]
        #inc are the selected cycles
        inc=np.flatnonzero(D11.cycle_stats.cycle_included_in_fit[this,:])
        #print "inc:" 
        #print inc
        plt.sca(self.ax2)
        plt.cla()
        # plot the heights and 
        yy=D11.cycle_stats.y_atc_mean[this,:].ravel()
        hh=D11.corrected_h.cycle_h_shapecorr[this,:].ravel()
        ss=D11.corrected_h.cycle_h_shapecorr_sigma[this,:].ravel()
        plt.errorbar(yy, hh, ss, fmt='o')
        plt.plot(yy, D11.cycle_stats.h_uncorr_mean[this,:].ravel(), 'kx')
        plt.plot(yy[inc], hh[inc],'r*', markersize =12)
        plt.sca(self.ax3)
        plt.cla()
        cycles=np.arange(yy.size)
        plt.errorbar(cycles.ravel(), hh.ravel(), ss.ravel(), fmt='o')
        plt.plot(cycles[inc], hh[inc],'r*', markersize=12)
        plt.show(block=False)         
        return
              

  