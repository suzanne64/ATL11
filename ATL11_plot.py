# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:05:50 2018

@author: ben
"""
import matplotlib.pyplot as plt
import numpy as np

class ATL11_plot:
    def __init__(self, D11, P11_list):
        self.D11=D11
        self.P11=P11_list
        fig=plt.figure()
        
        self.ax2=plt.axes([0.05,  0.05, 0.4, 0.375])
        self.ax3=plt.axes([0.55,  0.05, 0.4, 0.375])
        self.ax1=plt.axes([0.05, 0.525, 0.9, 0.375])
   
        self.h_errorbars=D11.plot()

        fig.canvas.mpl_connect('pick_event', self.pick_event)
        plt.show(block=True)
        return
        
    def pick_event(self, event):
         
        xx=event.artist.get_xdata()
        ii=event.ind
        x0=[temp.ref_surf.ref_pt_x_atc for temp in self.P11]
        this=np.flatnonzero(x0==xx[ii])
        print("this is %d" % this)
        D11=self.P11[this]
        print "x0=%d" % x0[this]
        rsp=np.flatnonzero(D11.pass_stats.pass_included_in_fit)
        print "rsp:" 
        print rsp
        plt.sca(self.ax2)
        plt.cla()
        yy=D11.pass_stats.y_atc_mean.ravel()
        hh=D11.corrected_h.pass_h_shapecorr.ravel()
        ss=D11.corrected_h.pass_h_shapecorr_sigma.ravel()
        plt.errorbar(yy.ravel(), hh.ravel(), ss.ravel(), fmt='o')
        plt.plot(yy.ravel(), D11.pass_stats.h_uncorr_mean.ravel(), 'kx')

        
        print("yy.shape is %d" %yy.shape)
        print("yy.shape[rsp] is %d " % yy[rsp].shape)
        plt.plot(yy[rsp], hh[rsp],'r*',markersize =12)
        plt.sca(self.ax3)
        plt.cla()
        passes=np.arange(yy.size)
        print passes
        plt.errorbar(passes.ravel(), hh.ravel(), ss.ravel(), fmt='o')
        plt.plot(passes[rsp], hh[rsp],'r*', markersize=12)
        plt.show(block=False)         
        return
              

  