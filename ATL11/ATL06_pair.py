# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:27:00 2017

@author: ben
"""
import numpy as np

class ATL06_pair:
    def __init__(self, D6=None, pair_data=None):
        if D6 is not None:
            #initializes based on input D6, assumed to contain one pair
            # 2a. Set pair_data x and y
            self.x=np.mean(D6.x_atc)  # mean of the pair, nan if not both defined
            self.y=np.mean(D6.y_atc)
            self.dh_dx=D6.dh_fit_dx
            self.dh_dx.shape=[1,2]
            self.dh_dy=np.mean(D6.dh_fit_dy)
            self.dh_dy_sigma=np.sqrt(np.sum(D6.h_li_sigma**2))/np.abs(np.diff(D6.y_atc))
            self.delta_time=np.mean(D6.delta_time)
            self.segment_id=np.mean(D6.segment_id)
            self.cycle=np.mean(D6.cycle_number)
            self.h=D6.h_li
            self.h.shape=[1,2]
            self.valid=np.zeros(1, dtype='bool')
        elif pair_data is not None:
            # initializes based on a list of pairs, to produce a structure with numpy arrays for fields
            for field in ('x','y','dh_dx','dh_dy','delta_time','segment_id','cycle','h','valid'):
                setattr(self, field, np.c_[[getattr(this_pair,field).ravel() for this_pair in pair_data]])
        else:
            #initializes an empty structure
            for field in ('x','y','dh_dx','dh_dy','delta_time','segment_id','cycle','h','valid'):
                setattr(self, field, np.NaN)

    def __getitem__(self, key):
        temp06=ATL06_pair()
        for field in ('x','y','dh_dx','dh_dy','delta_time','segment_id','cycle','h','valid'):
            temp_field=getattr(self, field)
            if len(temp_field.shape)>1 and temp_field.shape[1] > 1:
                setattr(temp06, temp_field[key,:])
            else:
                setattr(temp06, temp_field[key])
        return temp06

    def from_ATL06(self, D6, datasets=None):
        pair_list=list()
        for i in np.arange(D6.shape[0]):
            pair_list.append(ATL06_pair(D6=D6.copy_subset(i, by_row=True, datasets=datasets)))
        self=ATL06_pair(pair_data=pair_list)
        return self
    