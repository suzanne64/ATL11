# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:28:30 2018

@author: ben
"""
import h5py
import numpy as np
from osgeo import osr
 

class point_data:
    np.seterr(invalid='ignore')
    def __init__(self, list_of_fields=list(), SRS_proj4=None, columns=1):
        self.list_of_fields=list_of_fields
        self.SRS_proj4=SRS_proj4
        self.columns=1

    def copy_attrs(self):
        return point_data(list_of_fields=self.list_of_fields, SRS_proj4=self.SRS_proj4, columns=self.columns)
    
    def from_file(self, filename, field_dict=None, index_range=(0, -1)):
        h5_f=h5py.File(filename, 'r')
        nan_fields=list()
        for group in field_dict.keys():
            for field in field_dict[group]:
                if field not in self.list_of_fields:
                    self.list_of_fields.append(field)
                try:
                    if group is None:
                        if self.columns==0 or self.columns is None:
                            setattr(self, field, np.array(h5_f[field][index_range[0]:index_range[1]]).transpose())
                        else:
                            setattr(self, field, np.array(h5_f[field][:,index_range[0]:index_range[1]]).transpose())
                    else:
                        if self.columns==0 or self.columns is None:
                            setattr(self, field, np.array(h5_f[group][field][index_range[0]:index_range[1]]).transpose())  
                        else:
                            setattr(self, field, np.array(h5_f[group][field][:,index_range[0]:index_range[1]]).transpose())  
                except KeyError:
                    nan_fields.append(field)            
            # find the first populated field
        if len(nan_fields) > 0:
            for field in self.list_of_fields:
                if hasattr(self, field):
                    self.shape=getattr(self, field).shape
                    break
            if self.shape is not None:
                for field in nan_fields:
                    setattr(self, field, np.zeros(self.shape)+np.NaN)
        h5_f.close()
        return self

    
    def get_xy(self, proj4_str=None, EPSG=None):
        out_srs=osr.SpatialReference()
        if proj4_str is None and EPSG is not None:
            out_srs.ImportFromProj4(EPSG)
        else:
            out_srs.ImportFromProj4(proj4_str)
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        #x, y, z = list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(D.longitude), np.ravel(D.latitude), np.zeros_like(np.ravel(D.latitude)))]))
        if self.latitude.size==0:
            self.x=np.zeros_like(self.latitude)
            self.y=np.zeros_like(self.latitude)
        else:
            x, y, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(self.longitude), np.ravel(self.latitude), np.zeros_like(np.ravel(self.latitude)))]))
            self.x=np.reshape(x, self.latitude.shape)
            self.y=np.reshape(y, self.longitude.shape)
        if 'x' not in self.list_of_fields:
            self.list_of_fields.append('x')
            self.list_of_fields.append('y')
        return self
    
    def append(self, D):
        for field in self.list_of_fields:
            setattr(self, np.c_[getattr(self, field), getattr(D, field)])
        return self
    
    def from_dict(self, dd, list_of_fields=None):
        if list_of_fields is not None:
            self.list_of_fields=list_of_fields
        else:
            self.list_of_fields=dd.keys()
        for field in self.list_of_fields:
                setattr(self, field, dd[field])
        return self

    def from_list(self, D_list):
        try:
            for field in self.list_of_fields:
                data_list=[getattr(this_D, field) for this_D in D_list]       
                setattr(self, field, np.concatenate(data_list, 0))
        except TypeError:
            for field in self.list_of_fields:
                setattr(self, field, getattr(D_list, field))
        return self
    
    def index(self, index):
        for field in self.list_of_fields:
            setattr(self, field, getattr(self, field)[index])
        return self
        
    def subset(self, index, by_row=True, datasets=None):
        dd=dict()
        if self.columns is not None and self.columns >=1 and by_row is not None:
            by_row=True
        if datasets is None:
            datasets=self.list_of_fields
        for field in datasets:
            temp_field=self.__dict__[field]
            if temp_field.ndim ==1:
                dd[field]=temp_field[index]
            else:
                if by_row is not None and by_row:
                    dd[field]=temp_field[index,:]
                else:
                    dd[field]=temp_field.ravel()[index]
        return self.copy_attrs().from_dict(dd, list_of_fields=datasets)

