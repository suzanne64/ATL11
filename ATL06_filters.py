# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:47:16 2019

@author: ben
"""
import numpy as np

def phDensityFilter(D6, minDensity={'weak':1, 'strong':4}, setValid=True, toNaN=False, subset=False):
    mask=np.zeros_like(D6.n_fit_photons, dtype=bool)
    
    for beam in [0, 1]:
        phDensity=D6.n_fit_photons[:,beam]/D6.w_surface_window_final[:,beam]
        mask[np.isfinite(phDensity), beam]=phDensity[np.isfinite(phDensity)] > minDensity[D6.beam_type[beam]]

    if setValid:
        D6.valid=D6.valid & mask
    
    if toNaN:
        D6.h_li[mask==0]=np.NaN
    if subset:
        D6.index(np.any(mask==1, axis=1))

    return mask

def segDifferenceFilter(D6, tol=2, setValid=True, toNaN=False, subset=False):
    dAT=20.
    if D6.h_li.shape[0] < 3:
        mask=np.ones_like(D6.h_li, dtype=bool)
        return mask
    EPplus=D6.h_li + dAT*D6.dh_fit_dx
    EPminus=D6.h_li - dAT*D6.dh_fit_dx
    segDiff=np.zeros_like(D6.h_li)
    segDiff[0:-1,:]=np.abs(EPplus[0:-1,:]-D6.h_li[1:, :])
    segDiff[1:,:]=np.maximum(segDiff[1,:], np.abs(D6.h_li[0:-1,:]-EPminus[1:,:]))
    
    mask=segDiff<tol
    if setValid:
        D6.valid=D6.valid & mask
    if toNaN:
        D6.h_li[mask==0]=np.NaN
    if subset:
        D6.index(np.all(mask==1, axis=1))

    return mask
    
def qualitySummary(D6, includeDensity=False, setValid=True, includeSigSource=False, toNaN=False, subset=False):
    mask =( D6.h_robust_sprd < 1 ) & \
          ( D6.h_li_sigma < 1 ) & \
          ( D6.snr_significance < 0.02)
    if includeSigSource:
        mask = mask & (D6.signal_selection_source <= 1)
    if includeDensity:
        mask = mask & phDensityFilter(D6, setValid=False)
        
    atl06QualitySummary = mask==0
    if setValid:
        D6.valid = D6.valid & atl06QualitySummary==0
    if toNaN: 
        D6.h_li[atl06QualitySummary > 0]=np.NaN
    if subset:
        D6.subset(np.all(atl06QualitySummary, axis=1))

    return atl06QualitySummary
