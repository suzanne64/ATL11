#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:31:29 2019

@author: ben
"""
import numpy as np

class group(object):
    # Class to contain an ATL11 structure
    # in ATL11 groups, some datasets have one value per reference point (per_pt_fields)
    # some have one value for each reference point and each cycle (full_fields)
    # some have one value for each point and each polynomial coefficient (poly_fields)
    # all of these are initialized to arrays of the appropriate size, filled with NaNs

    def __init__(self, N_pts, cycles, N_coeffs, per_pt_fields=None, full_fields=None, poly_fields=None, xover_fields=None):
        # input variables:
        #  N_pts: Number of reference points to allocate
        #  cycles: cycles for which to allocate data
        #  N_coeffs: Number of polynomial coefficients to allocate
        #  per_pt_fields: list of fields that have one value per reference point
        #  full_fields: list of fields that have one value per cycle per reference point
        #  poly_fields: list of fields that have one value per polynomial degree combination per reference point

        # assign fields of each type to their appropriate shape and size
        if per_pt_fields is not None:
            for field in per_pt_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, 1]))
        if full_fields is not None:
            for field in full_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, cycles[1]-cycles[0]+1]))
        if poly_fields is not None:
            for field in poly_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, N_coeffs]))
        if xover_fields is not None:
            for field in xover_fields:
                setattr(self, field, [])

        # assemble the field names into lists:
        self.per_pt_fields=per_pt_fields
        self.full_fields=full_fields
        self.poly_fields=poly_fields
        self.xover_fields=xover_fields
        self.list_of_fields=self.per_pt_fields+self.full_fields+self.poly_fields+self.xover_fields
        self.attrs=dict()
        
    def __repr__(self):
        out=''
        for field in self.list_of_fields:
            out += field+':\n'
            out += str(getattr(self, field))
            out += '\n'
        return out

    def index(self, ind, N_cycles=None, N_coeffs=None, xover_ind=None):
        """
        index an ATL11 data object.
        """
        if N_coeffs is None:
            N_coeffs=self.N_coeffs
        if N_cycles is None:
            N_cycles=self.N_cycles
        try:
            N_pts=len(ind)
        except TypeError:
            N_pts=1

        target=group(N_pts, N_cycles, N_coeffs, per_pt_fields=self.per_pt_fields.copy(), full_fields=self.full_fields.copy(), poly_fields=self.poly_fields.copy(), xover_fields=self.xover_fields.copy())
                # assign fields of each type to their appropriate shape and size
        if self.per_pt_fields is not None:
            for field in self.per_pt_fields:
                setattr(target, field, getattr(self, field)[ind])
        if self.full_fields is not None:
            for field in self.full_fields:
                setattr(target, field, getattr(self, field)[ind, :])
        if self.poly_fields is not None:
            for field in self.poly_fields:
                setattr(target, field,  getattr(self, field)[ind, :])
        if self.xover_fields is not None:
            # need to pick out the matching crossover-point fields
            for field in self.xover_fields:
                if xover_ind is not None:
                    try:
                        setattr(target, field, getattr(self, field)[xover_ind])
                    except IndexError:
                        setattr(target, field, np.array([]))
                else:
                    setattr(target, field, [])
        if hasattr(self,'x'):
            self.x=self.x[ind]
            self.y=self.y[ind]
        return target