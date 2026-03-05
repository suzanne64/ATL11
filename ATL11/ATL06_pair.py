# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:27:00 2017

@author: ben
"""
import numpy as np
import pointCollection

class ATL06_pair(pointCollection.data):
    def __init__(self, D6=None, pair_data=None):
        if D6 is not None:
            shp = (D6.h_li.shape[0], 1)
            #initializes based on input D6, assumed to contain one pair
            # 2a. Set pair_data x and y
            self.x=np.mean(D6.x_atc, axis=1).reshape(shp)  # mean of the pair, nan if not both defined
            self.y=np.mean(D6.y_atc, axis=1).reshape(shp)
            self.dh_dx=D6.dh_fit_dx
            self.dh_dy=np.mean(D6.dh_fit_dy, axis=1).reshape(shp)
            self.dh_dy_sigma=np.sqrt(np.sum(D6.h_li_sigma**2))/np.abs(np.diff(D6.y_atc))
            self.delta_time=np.mean(D6.delta_time, axis=1).reshape(shp)
            self.segment_id=np.mean(D6.segment_id, axis=1).reshape(shp)
            self.cycle=np.mean(D6.cycle_number, axis=1).reshape(shp)
            self.h=D6.h_li
            self.valid=np.zeros(shp, dtype='bool')
        elif pair_data is not None:
            # initializes based on a list of pairs, to produce a structure with numpy arrays for fields
            for field in ('x','y','dh_dx','dh_dy','delta_time','segment_id','cycle','h','valid'):
                setattr(self, field, np.c_[[getattr(this_pair,field).ravel() for this_pair in pair_data]])
        else:
            #initializes an empty structure
            for field in ('x','y','dh_dx','dh_dy','delta_time','segment_id','cycle','h','valid'):
                setattr(self, field, np.nan)
