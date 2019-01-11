# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:44:31 2018

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
def pt_blockmedian(x, y, z, delta, xy0=[0.,0.], return_index=False):
    yscale=np.ceil((np.max(y)-np.min(y))/delta*1.1)
    zscale=(np.max(z)-np.min(z))*1.1
    xr=np.floor((x-xy0[0])/delta)
    yr=np.floor((y-xy0[1])/delta)
    xyind=xr*yscale+(yr-np.min(yr))
    ind=np.argsort(xyind+(z-np.min(z))/zscale)
    xs=x[ind]
    ys=y[ind]
    zs=z[ind]
    xyind=xyind[ind]
    ux, ix=np.unique(xyind, return_index=True)
    xm=np.zeros_like(ux)+np.NaN
    ym=np.zeros_like(ux)+np.NaN
    zm=np.zeros_like(ux)+np.NaN
    if return_index:
        ind=np.zeros((ux.size,2), dtype=int)
    ix=np.concatenate(ix, xyind.size)
    for  count, i0 in enumerate(ix[:-1]):
        ii=np.arange(i0, ix[count+1], dtype=int)
        iM=ii.size/2.-1
        if iM-np.floor(iM)==0:
            iM=int(np.floor(iM))
            iM=ii[[iM, iM+1]]
            xm[count]=(xs[iM[0]]+xs[iM[1]])/2.
            ym[count]=(ys[iM[0]]+ys[iM[1]])/2.
            zm[count]=(zs[iM[0]]+zs[iM[1]])/2.
            if return_index:
                ind[count,:]=iM
        else:
            iM=ii[int(iM)]
            xm[count]=xs[iM]
            ym[count]=ys[iM]
            zm[count]=zs[iM]
            if return_index:
                ind[count,:]=iM
        #plt.figure(1); plt.clf(); plt.subplot(211); plt.cla(); plt.scatter(xs[ii]-xm[count], ys[ii]-ym[count], c=zs[ii]); plt.axis('equal'); plt.subplot(212); plt.plot(zs[ii]); 
        #plt.pause(1)
        #print(count)
    if return_index:
        return xm, ym, zm, ind
    else:
        return xm, ym, zm

