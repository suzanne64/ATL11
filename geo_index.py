# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:58:19 2018

@author: ben

This is a class that lets us generate a coarse-resolution database of the 
data-point locations in a point-data file, and to build hierarchical indices
to allow efficient searches for data.

"""
import numpy as np
import h5py
from osgeo import osr
import matplotlib.pyplot as plt
from ATL06_data import ATL06_data        
       
class geo_index(dict):
    def __init__(self, delta=[1000,1000], SRS_proj4=None):
        dict.__init__(self)
        self.attrs={'delta':delta,'SRS_proj4':SRS_proj4, 'n_files':0}
        self.h5_file=None
        
    def __del__(self):
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
        # build a geo_index from a list of x, y points, for a specified filename
        # and file_type.  If the file_type is 'geo_index', optionally sepecify a
        # value for 'fake_offset_val'
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
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'offset_start':np.array(first[ind], ndmin=1), 'offset_end':np.array(last[ind], ndmin=1)}
            else:
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'offset_start':np.array(fake_offset_val, ndmin=1), 'offset_end':np.array(fake_offset_val, ndmin=1)}

        self.attrs['file_0']=filename
        self.attrs['type_0']=file_type
        self.attrs['n_files']=1
        return self
          
    def from_list(self, index_list, delta=None, SRS_proj4=None, copy_first=False): 
        # build a geo_index from a list of geo_indices.  
        # Each bin in the resulting geo_index contains information for reading
        # the files indexed by the geo_indices in index_list
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
                    for field in ('offset_start','offset_end'):
                        append_data(self[bin], field, index[bin][field][keep])
                    
                else:
                    self[bin]=dict()
                    self[bin]['file_num']=newFileNums
                    for field in ('offset_start','offset_end'):
                        self[bin][field]=index[bin][field][keep]
        self.attrs['n_files']=len(fileListTo)
        return self

    def from_file(self, index_file, read_file=True):
        # read geo_index info from file 'index_file.'  
        # If read_file is set to False, the file is not read, but the 
        # h5_file_index attribute of the resulting geo_index is set to a 
        # reference to the hdf_file's 'index' attribute.  This seems to be
        # faster than reading the whole file.
        h5_f = h5py.File(index_file,'r')
        h5_i = h5_f['index']
        if read_file:
            for bin in h5_i.keys():
                self[bin]=h5_i[bin]
        self.attrs=h5_i.attrs
        self.h5_file=h5_f
        self.h5_file_index=h5_f['index']
        return self
        
    def to_file(self, filename):
        # write the current geoindex to hdf5 file 'filename'
        indexF=h5py.File(filename,'a') 
        if 'index' in indexF:
            del indexF['index']
        indexGrp=indexF.create_group('index')
        indexGrp.attrs['n_files'] = 0
        indexGrp.attrs['delta'] = self.attrs['delta']
        indexGrp.attrs['SRS_proj4'] = self.attrs['SRS_proj4']
        for key in self.keys():
            indexGrp.create_group(key)
            for field in ['file_num','offset_start','offset_end']:
                indexGrp[key].create_dataset(field,data=self[key][field])
        for ii in range(self.attrs['n_files']):
            this_key='file_%d' % ii
            indexGrp.attrs[this_key]=self.attrs[this_key]
            this_type='type_%d' % ii
            indexGrp.attrs[this_type]=self.attrs[this_type]
        indexF.close()
        return

    def query_latlon(self, lat, lon, get_data=True, fields=None):
        # query the current geo_index for all bins that match the bin locations
        # provided in (lat, lon),  Optionally return data, with field query in 'fields'
        out_srs=osr.SpatialReference()
        out_srs.ImportFromProj4(self.attribs['SRS_proj4'])
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        x, y = list(zip(*[ct.TransformPoint(xy) for xy in zip(np.ravel(lon), np.ravel(lat))]))
        delta=self.attribs['delta']
        xb=np.round(x/delta[0])*delta[0]
        yb=np.round(y/delta[1])*delta[1]
        return self.query_xy(xb, yb, get_data=get_data, fields=fields)

    def query_xy_box(self, xr, yr, get_data=True, fields=None):
        # query the current geo_index for all bins in the box specified by box [xr,yr]
        xy_bin=self.bins_as_array() 
        these=np.logical_and(np.logical_and(xy_bin[:,0] >= xr[0], xy_bin[:,0] <= xr[1]), 
            np.logical_and(xy_bin[:,1] >= yr[0], xy_bin[:,1] <= yr[1]))
        return self.query_xy(xy_bin[these,0], xy_bin[these,1], get_data=get_data)
    
    def bins_as_array(self):
        if len(self)>0:
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in self.keys()]]
        else:
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in self.h5_file_index.keys()]]
        return xy_bin
 
    def query_xy(self, xb, yb, delta=None, cleanup=True, get_data=True, fields=None, pad=None, dir_root=None):
        # check if data exist within the current geo index for bins in lists/arrays
        #     xb and yb.  
        # If argument delta is provided, find the bins in the current geo_index
        #     that round to (xb, yb)
        # If 'delta' is provided, read the underlying data sources, possibly recursively
        #    otherwise return a query_result: a dict with one entry for each source file 
        #    in the current geo_index, giving the bin locations provided by that file, 
        #    and the offsets in the file corresponding to each.
        # If 'pad' is provided, include bins between xb-pad*delta and xp+pad*delta (inclusive)
        #     in the query (likewise for y)
        if delta is None:
            delta=self.attrs['delta']
        if isinstance(xb, np.ndarray):
            xb=xb.copy().ravel()
            yb=yb.copy().ravel()
        if pad is not None:
            [xp,yp]=np.meshgrid(np.arange(-pad, pad+1)*delta[0], np.arange(-pad, pad+1)*delta[1])
            xp=xp.ravel(); yp=yp.ravel();
            xb=np.concatenate([xpi+xb for xpi in xp]).ravel()
            yb=np.concatenate([ypi+yb for ypi in yp]).ravel()
            # keep only the unique members of xb and yb
            xb, yb = unique_points(xb, yb, delta)

        # make a temporary geo_index to hold the subset of the current geoindex
        # corresponding to xb and yb
        temp_gi=geo_index(delta=self.attrs['delta'], SRS_proj4=self.attrs['SRS_proj4'])
        if (np.array(delta)==np.array(self.attrs['delta'])).all():  
            # query self at its native resolution
            for bin in set(zip(xb, yb)):
               bin_name='%d_%d' % bin
               if bin_name in self:
                   temp_gi[bin_name]=self[bin_name]
               elif hasattr(self, 'h5_file_index') and bin_name in self.h5_file_index:
                   temp_gi[bin_name]=self.h5_file_index[bin_name]
        else:
            # query self at a different (coarser?) resolution: need to query bins at the finer resolution
            self_delta=self.attrs['delta']
            for bin in set(zip(xb, yb)):
                x0=self_delta[0]*np.round(np.arange(bin[0]-delta[0]/2, bin[0]+delta[0]/2, self_delta[0])/self_delta[0]) 
                y0=self_delta[1]*np.round(np.arange(bin[1]-delta[1]/2, bin[1]+delta[1]/2, self_delta[1])/self_delta[0]) 
                for bin_x in x0:
                    for bin_y in y0:
                        bin_name=u'%d_%d' % (bin_x, bin_y)
                        if bin_name in self:
                            temp_gi[bin_name]=self[bin_name]
                        elif hasattr(self, 'h5_file_index') and bin_name in self.h5_file_index:
                            temp_gi[bin_name]=self.h5_file_index[bin_name]  
        if len(temp_gi.keys())==0:
            return None        
        temp_dict=dict()
        for field in ['file_num','offset_start','offset_end']:
           temp_dict[field]=np.concatenate([temp_gi[key][field] for key in sorted(temp_gi)])
        # build an array of x and y values for the bins in temp_gi
        xy0=np.concatenate([np.tile(np.fromstring(key, sep='_').astype(int),(temp_gi[key]['file_num'].size,1)) for key in sorted(temp_gi)], axis=0)
        out_file_nums=np.unique(temp_dict['file_num'])
        query_results=dict()
        for out_file_num in out_file_nums:
            these=temp_dict['file_num']==out_file_num
            i0=np.array(temp_dict['offset_start'][these], dtype=int)
            i1=np.array(temp_dict['offset_end'][these], dtype=int)
            xy=xy0[these,:]
            if cleanup:
                # clean up the output: when the start of the next segment is 
                #within 1 of the end of the previous, stick them together  
                ii=np.argsort(i0)
                i0=i0[ii]
                i1=i1[ii]
                xy=xy[ii,:]
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
                xy=xy[keep,:]
            query_results[self.attrs['file_%d' % out_file_num]]={
            'type':self.attrs['type_%d' % out_file_num],
            'offset_start':i0,
            'offset_end':i1,
            'x':xy[:,0],
            'y':xy[:,1]}  
        if get_data:
             query_results=get_data_for_geo_index(query_results, delta=self.attrs['delta'], fields=fields, dir_root=dir_root)
        return query_results

def get_data_for_geo_index(query_results, delta=None, fields=None, dir_root=None):
    # read the data from a set of query results  
    # Currently the function knows how to read h5_geoindex and ATL06 data.  
    # Append more cases as needed
    if dir_root is not None:
        dir_root += '/'
    data=list()
    for file_key, result in query_results.items():
        if result['type'] == 'h5_geoindex':
            data += geo_index().from_file(dir_root+file_key).query_xy(result['x'], result['y'], delta=delta, fields=fields, get_data=True)
        if result['type'] == 'ATL06':
            if fields is None:
                fields={None:(u'latitude',u'longitude',u'h_li',u'delta_time')}
            D6_file, pair=file_key.split(':pair')             
            D6=[ATL06_data(filename=dir_root+D6_file, beam_pair=int(pair), index_range=np.array(temp), NICK=True, field_dict=fields) for temp in zip(result['offset_start'], result['offset_end'])]           
            if isinstance(D6,list):
                data += D6  
            else:
                data.append(D6)
    return data

def unique_points(x, y, delta=[1, 1]):
    xyb=np.concatenate([np.array(xybi).reshape([1,2]) for xybi in set(zip(np.round(np.array(x)/delta[0])*delta[0], np.round(np.array(y)/delta[0])*delta[0]))], axis=0)
    return xyb[:,0], xyb[:,1]

def append_data(group, field, newdata):  
    # utility function that can append data either to an hdf5 field or a dict of numpy array
    try:
        old_shape=np.array(group[field].shape)
        new_shape=old_shape.copy()
        new_shape[0]+=newdata.shape[0]
        group[field].reshape((new_shape))
        group[field][old_shape[0]:new_shape[0],:]=newdata
    except:
        group[field]=np.concatenate((group[field], newdata), axis=0)
    return

def index_list_for_files(filename_list, file_type, delta, SRS_proj4, dir_root=None):
    index_list=list()
    out_srs=osr.SpatialReference()
    out_srs.ImportFromProj4(SRS_proj4)
    ll_srs=osr.SpatialReference()
    ll_srs.ImportFromEPSG(4326)
    ct=osr.CoordinateTransformation(ll_srs, out_srs)
    for number, filename in enumerate(filename_list):
        if dir_root is not None:
            # eliminate the string in 'dir_root' from the filename
            filename=filename.replace(dir_root,'')
        if file_type in ['ATL06']:   
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
            # read the file as a collection of points
            temp_gi=geo_index().from_file(filename)
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in temp_gi.keys()]] 
            index_list.append(geo_index(delta=delta, SRS_proj4=SRS_proj4).from_xy(xy_bin[:,0], xy_bin[:,1], filename, file_type, number=0, fake_offset_val=-1))
                 
    return index_list