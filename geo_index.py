# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:58:19 2018

@author: ben
"""
import numpy as np
import h5py
from osgeo import osr
from ATL06_data import ATL06_data        
  
def append_data(group, field, newdata):    
    try:
        old_shape=np.array(group[field].shape)
        new_shape=old_shape.copy()
        new_shape[0]+=newdata.shape[0]
        group[field].reshape((new_shape))
        group[field][old_shape[0]:new_shape[0],:]=newdata
    except:
        group[field]=np.concatenate((group[field], newdata), axis=0)
      

class geo_index(dict):
    def __init__(self, delta=[1000,1000], SRS_proj4=None):
        dict.__init__(self)
        self.attrs={'delta':delta,'SRS_proj4':SRS_proj4, 'n_files':0}
        self.h5_file=None
    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
        return

    def __copy__(self):
        out=dict()
        out.attrs=self.attrs.copy()
        for field in self.keys():
            out[field]=self[field].copy()
        return out
        
    def from_xy(self, x, y, filename, file_type, number=0, fake_offset_val=None):    
        delta=self.attrs['delta']
        x_bin=np.round(x/delta[0])
        y_bin=np.round(y/delta[1])
        
        # sort by magnitude of (x,y), then by angle
        ordering=np.sqrt(x_bin**2+y_bin**2)+(np.arctan2(y_bin, x_bin)+np.pi)/2/np.pi
        uOrd, first=np.unique(ordering, return_index=True)
        uOrd, temp=np.unique(-ordering[::-1], return_index=True)
        last=len(ordering)-1-temp[::-1]
     
        for ind in range(len(first)):
            key='%d_%d' % (delta[0]*x_bin[first[ind]], delta[1]*y_bin[first[ind]])
            if fake_offset_val is None:
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'ind_start':np.array(first[ind], ndmin=1), 'ind_end':np.array(last[ind], ndmin=1)}
            else:
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'ind_start':np.array(fake_offset_val, ndmin=1), 'ind_end':np.array(fake_offset_val, ndmin=1)}

        self.attrs['file_0']=filename
        self.attrs['type_0']=file_type
        self.attrs['n_files']=1
        return self
          
    def from_list(self, index_list, delta=None, SRS_proj4=None, copy_first=False): 
        if copy_first:
            self=index_list[0].copy
        else:
            self=index_list[0]
        if len(index_list)==0:
            return

        fileListTo=[self.attrs['file_%d' % fileNum] for fileNum in range(self.attrs['n_files'])]           
            
        for index in index_list[1:]:
            # check if a particular input file is alread in the output index, otherwise add it
            # keep track of how the filenames in the input index correspond to those in the output index
            num_out=dict()
            alreadyIn=list()
            for fileNum in range(index.attrs['n_files']):
                thisFileName=index.attrs['file_%d' % fileNum]
                thisFileType=index.attrs['type_%d' % fileNum]
                if thisFileName not in fileListTo:
                    fileListTo.append(thisFileName)
                    self.attrs['file_%d' % (len(fileListTo)-1)] = thisFileName 
                    self.attrs['type_%d' % (len(fileListTo)-1)] = thisFileType
                else:
                    alreadyIn.append(fileNum)
                num_out[fileNum]=fileListTo.index(thisFileName)
            for bin in index.keys():       
                newFileNums=index[bin]['file_num'].copy()
                keep=np.logical_not(np.in1d(newFileNums, alreadyIn))
                if not np.any(keep):
                    continue
                newFileNums=newFileNums[keep]
                for row in range(newFileNums.shape[0]):
                    newFileNums[row]=num_out[newFileNums[row]]
                if bin in self:
                    #append_data(self[bin]['file_num'], newFileNums)
                    append_data(self[bin],'file_num', newFileNums)
                    append_data(self[bin],'ind_start', index[bin]['ind_start'][keep])
                    append_data(self[bin],'ind_end', index[bin]['ind_end'][keep])
                else:
                    self[bin]=dict()
                    self[bin]['file_num']=newFileNums
                    self[bin]['ind_start']=index[bin]['ind_start'][keep]
                    self[bin]['ind_end']=index[bin]['ind_end'][keep]
                   
        self.attrs['n_files']=len(fileListTo)
        return self

    def from_file(self, index_file):
        h5_f = h5py.File(index_file,'r')
        h5_i = h5_f['index']
        for bin in h5_i.keys():
            self[bin]=h5_i[bin]
        self.attrs=h5_i.attrs
        self.h5_file=h5_f
        return self
        
    def to_file(self, filename):
        indexF=h5py.File(filename,'a') 
        if 'index' in indexF:
            del indexF['index']
        indexGrp=indexF.create_group('index')
        indexGrp.attrs['n_files'] = 0
        indexGrp.attrs['delta'] = self.attrs['delta']
        indexGrp.attrs['SRS_proj4'] = self.attrs['SRS_proj4']
        for key in self.keys():
            indexGrp.create_group(key)
            for field in ['file_num','ind_start','ind_end']:
                indexGrp[key].create_dataset(field,data=self[key][field])
        for ii in range(self.attrs['n_files']):
            this_key='file_%d' % ii
            indexGrp.attrs[this_key]=self.attrs[this_key]
            this_type='type_%d' % ii
            indexGrp.attrs[this_type]=self.attrs[this_type]
        indexF.close()
        return

    def query_latlon(self, lat, lon):
        out_srs=osr.SpatialReference()
        out_srs.ImportFromProj4(self.attribs['SRS_proj4'])
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        x, y = list(zip(*[ct.TransformPoint(xy) for xy in zip(np.ravel(lat), np.ravel(lon))]))
        delta=self.attribs['delta']
        xb=np.round(x/delta[0])*delta[0]
        yb=np.round(y/delta[1])*delta[1]
        return self.query_xy(xb, yb)

    def query_xy(self, xb, yb):
        temp_gi=geo_index(delta=self.attrs['delta'], SRS_proj4=self.attrs['SRS_proj4'])
        for bin in set(zip(xb, yb)):
            bin_name='%d_%d' % bin
            if bin_name in self:
                temp_gi[bin_name]=self[bin_name]
        temp_dict=dict()
        for field in ['file_num','ind_start','ind_end']:
           temp_dict[field]=np.concatenate([temp_gi[key][field] for key in temp_gi.keys()])
        out_file_nums=np.unique(temp_dict['file_num'])
        out=dict()
        for out_file_num in out_file_nums:
            these=temp_dict['file_num']==out_file_num
            i0=np.array(temp_dict['ind_start'][these], dtype=int)
            i1=np.array(temp_dict['ind_end'][these], dtype=int)
            # cleanup the output: when the start of the next segment is within 
            #    1 of the end of the last, stick them together  
            ii=np.argsort(i0)
            i0=i0[ii]
            i1=i1[ii]
            keep=np.zeros(len(i0), dtype=bool)
            this=0
            keep[this]=True
            for kk in np.arange(1,len(i0)):
                if i0[kk]==i1[this]+1:
                    keep[kk]=False
                    i1[this]=i1[kk]
                else:
                    this=kk
                    keep[kk]=True
            i0=i0[keep]
            i1=i1[keep]
          
            out[self.attrs['file_%d' % out_file_num]]={
            'type':self.attrs['type_%d' % out_file_num],
            'ind_start':i0,
            'ind_end':i1}         
        return out

    def query_xy_box(self, xr, yr):
        xy_bin=np.c_[[np.fromstring(key, sep='_') for key in self.keys()]] 
        these=np.logical_and(np.logical_and(xy_bin[:,0] >= xr[0], xy_bin[:,0] <= xr[1]), 
            np.logical_and(xy_bin[:,1] >= yr[0], xy_bin[:,1] <= yr[1]))
        return self.query_xy(xy_bin[these,0], xy_bin[these,1])

def index_list_for_files(filename_list, file_type, delta, SRS_proj4):
    index_list=list()
    out_srs=osr.SpatialReference()
    out_srs.ImportFromProj4(SRS_proj4)
    ll_srs=osr.SpatialReference()
    ll_srs.ImportFromEPSG(4326)
    ct=osr.CoordinateTransformation(ll_srs, out_srs)
    if file_type in ['ATL06']:
        for number, filename in enumerate(filename_list):
            for beam_pair in (1, 2, 3):     
                D=ATL06_data(filename=filename, beam_pair=beam_pair, NICK=True, field_dict={None:('latitude','longitude','h_li','delta_time')})
                if D.latitude.shape[0] > 0:
                    x, y, z = list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(D.longitude), np.ravel(D.latitude), np.zeros_like(np.ravel(D.latitude)))]))
                    x=np.array(x).reshape(D.latitude.shape)
                    y=np.array(y).reshape(D.latitude.shape)                     
                    xp=np.nanmean(x, axis=1)
                    yp=np.nanmean(y, axis=1)
                    index_list.append(geo_index(delta=delta, SRS_proj4=SRS_proj4).from_xy(xp, yp, '%s:pair%d' % (filename, beam_pair), 'ATL06', number=number))
    if file_type in ['h5_geoindex']:
        for number, filename in filename_list:
            # read the file as a collection of points
            temp_gi=geo_index().from_file(filename)
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in temp_gi.keys()]] 
            index_list.append(geo_index(delta=delta, SRS_proj4=SRS_proj4).from_xy(xy_bin[:,0], xy_bin[:,1], filename, file_type, number=number, fake_offset=-1))
                 
    return index_list