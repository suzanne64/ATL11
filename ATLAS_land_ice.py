# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:50:06 2017

@author: ben
"""
 

def RDE(x):
    xs=x.copy()
    xs=xs(np.isfinite(xs))
    if len(xs)<2 :
        return np.nan
    ind=np.arange(0.5, len(xs))
    LH=np.interp(np.array([0.16, 0.84])*len(xs), ind, xs.sorted())
    return (LH[1]-LH[0])/2.

   
def flatten_struct(a):
    b=a[0].__init__
    if len(a)==1:
        b=a[0].copy()
        return b
    fields=a.dict().keys()
    for field in fields:
        b.setattr(field, np.concatenate([x.getattr(field) for x in a] ))
    return b
