#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:32:05 2019

@author: ben
"""
import numpy as np

class validMask(object):
    # class to hold validity flags for different attributes
    def __init__(self, dims, fields):
        for field in fields:
            setattr(self, field, np.zeros(dims, dtype='bool'))
    def __repr__(self):
        out=''
        for field in dir(self):
            if not field.startswith('__'):
                out += field+':\n'
                temp=getattr(self, field)
                out += str(temp)
                out += '\n'
        return out