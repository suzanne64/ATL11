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
        x0=[temp.x_atc_ctr for temp in self.P11]
        this=np.where(x0==xx[ii])[0]
        P11=self.P11[this]
        print "x0=%d" % x0[this]
        rsp= P11.ref_surf_passes.astype(int)-1
        plt.sca(self.ax2)
        plt.cla()
        plt.errorbar(P11.pass_y, P11.pass_h_shapecorr, P11.pass_h_shapecorr_sigma, fmt='o')
        plt.plot(P11.pass_y[rsp], P11.pass_h_shapecorr[rsp],'r*',markersize =12)
        plt.sca(self.ax3)
        plt.cla()
        plt.errorbar(np.arange(P11.pass_x.shape[0]), P11.pass_h_shapecorr, P11.pass_h_shapecorr_sigma, fmt='o')
        plt.plot(P11.ref_surf_passes-1, P11.pass_h_shapecorr[rsp],'r*', markersize=12)
        plt.show(block=False)         
        return
        

  