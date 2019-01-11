# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:58:19 2018

@author: ben

This is a class that lets us generate a coarse-resolution database of the 
data-point locations in a point-data file, and to build hierarchical indices
to allow efficient searches for data.
Coordinates below are always provided as tuples or lists, the first member of which is an array of x coordinates, the second is an array of y coordinates
"""
import numpy as np
import h5py
from osgeo import osr
import matplotlib.pyplot as plt
from ATL11.ATL06_data import ATL06_data        
from IS2_calval.Qfit_data import Qfit_data
from ATL11.point_data import point_data
       
class geo_index(dict):
    def __init__(self, delta=[1000,1000], SRS_proj4=None, data=None):
        dict.__init__(self)
        self.attrs={'delta':delta,'SRS_proj4':SRS_proj4, 'n_files':0}
        self.data=data
        if self.data is not None:
            if hasattr(data,'x'):
                self.from_xy([data.x, data.y])
            elif hasattr(data,'latitude'):
                self.from_latlon(self.data.latitude, self.data.longitude)
        self.h5_file=None
        
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
        return

    def __copy__(self):
        # copy method,
        out=geo_index()
        out.attrs=self.attrs.copy()
        for field in self.keys:
            out[field]=self[field].copy()     
        out.data=self.data
        return out    
    
    def copy_subset(self, xyBin=None, pad=None):
        # copy method, may specify which bins to copy
        out=geo_index()
        for attr in self.attrs:
            out.attrs[attr]=self.attrs[attr]
        #out.attrs=self.attrs.copy()
        if xyBin is None:
            these_keys=self.keys()
        else:
            if pad is not None and pad >0:
                xyBin=pad_bins(xyBin, pad, self.attrs['delta'])
            these_keys=self.keys_from_xy(xyBin)          
        for field in these_keys:
            if isinstance(self[field], dict):
                out[field]=self[field].copy()  
            else:
                if field not in out:
                    out[field]=dict()
                # this copies the reference to a h5 group
                for key in self[field].keys():
                    out[field][key]=self[field][key][:]
        out.data=self.data
        return out
        
    def reduce_delta(self, delta, data=None):
        # make a geo_index with a smaller delta value than the current
        if data is None:
            data=self.data
        out=geo_index(delta=delta)
        out.attrs=self.attrs.copy()
        out.attrs['delta']=delta
        for key in self.keys():
            i0=self[key]['offset_start']
            temp=geo_index().from_xy([data.x[i0:self[key]['offset_end']], data.y[i0:self[key]['offset_end']]])
            for key1 in temp:
                out[key]={'file_num':0, 'offset_start':i0+temp[key1]['offset_start'], 
                   'offset_end':i0+temp[key1]['offset_end']}
        return out 
        
    def from_xy(self, xy,  filename=None, file_type=None, number=0, fake_offset_val=None, first_last=None):  
        # build a geo_index from a list of x, y points, for a specified filename
        # and file_type.  If the file_type is 'geo_index', optionally sepecify a
        # value for 'fake_offset_val'
        delta=self.attrs['delta']
        xy_bin=np.round(np.c_[xy[0].ravel(), xy[1].ravel()]/delta)
        if first_last is None:
            # If the inputs haven't provided the first and last index for each bin, need to calculate it:
            # sort by magnitude of (x,y), then by angle
            ordering=np.sqrt(np.sum(xy_bin**2, axis=1))+(np.arctan2(xy_bin[:,0], xy_bin[:,1])+np.pi)/2/np.pi
            uOrd, first=np.unique(ordering, return_index=True)
            uOrd, temp=np.unique(-ordering[::-1], return_index=True)
            last=len(ordering)-1-temp[::-1]
            keys=['%d_%d' % (delta[0]*xy_bin[first[ind],0], delta[1]*xy_bin[first[ind],1]) for ind  in range(len(first))]
        else:
            # assume that the first_last values match the bins
            first, last=first_last
            keys=['%d_%d' % (xy_bin[ind,0]*delta[0], xy_bin[ind,1]*delta[1]) for ind in range(first.size)]
                        
        for ind, key in enumerate(keys):
            if fake_offset_val is None:
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'offset_start':np.array(first[ind], ndmin=1), 'offset_end':np.array(last[ind], ndmin=1)}
            else:
                self[key] = {'file_num':np.array(int(number), ndmin=1), 'offset_start':np.array(fake_offset_val, ndmin=1), 'offset_end':np.array(fake_offset_val, ndmin=1)}

        self.attrs['file_0']=filename
        self.attrs['type_0']=file_type
        self.attrs['n_files']=1
        return self
    
    def from_latlon(self, lat, lon,  filename=None, file_type=None, number=0, fake_offset_val=None):
        out_srs=osr.SpatialReference()
        out_srs.ImportFromProj4(self.attrs['SRS_proj4'])
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs).TransformPoint
        #xy=[ct(*xyz)[0:2] for xyz in zip(np.ravel(lon), np.ravel(lat), np.zeros_like(lat).ravel())]
        x, y, z = list(zip(*[ct(*xy) for xy in zip(np.ravel(lon), np.ravel(lat), np.zeros_like(lat).ravel())]))
        return self.from_xy([np.array(x),np.array(y)], filename, file_type, number, fake_offset_val)
    
    def from_list(self, index_list, delta=None, SRS_proj4=None, copy_first=False): 
        # build a geo_index from a list of geo_indices.  
        # Each bin in the resulting geo_index contains information for reading
        # the files indexed by the geo_indices in index_list
        if len(index_list)==0:
            return
        if copy_first:
            self=index_list[0].copy()
        else:
            self.attrs=index_list[0].attrs
            for bin_name in index_list[0]:
                self[bin_name]=index_list[0][bin_name]
                    
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
        #if read_file is True:
        #    h5_f.close()
        return self
        
    def to_file(self, filename):
        # write the current geoindex to h5 file 'filename'
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

    def for_file(self, filename, file_type, delta, SRS_proj4, number=0, dir_root=''):
        # make a geo_index for file 'filename'
        if dir_root is not None:
            # eliminate the string in 'dir_root' from the filename
            filename=filename.replace(dir_root,'')
        if file_type in ['ATL06']: 
            temp=list()
            this_field_dict={None:('latitude','longitude','h_li','delta_time')}
            for beam_pair in (1, 2, 3):     
                D=ATL06_data(beam_pair=beam_pair, field_dict=this_field_dict).from_file(filename).get_xy(SRS_proj4)
                if D.latitude.shape[0] > 0:
                    temp.append(geo_index(delta=delta, SRS_proj4=SRS_proj4).from_xy([np.nanmean(D.x, axis=1), np.nanmean(D.y, axis=1)], '%s:pair%d' % (filename, beam_pair), 'ATL06', number=number))
            self.from_list(temp)
        if file_type in ['ATM_Qfit']:
            D=Qfit_data(filename=filename, field_dict={'latitiude','longitude', 'time'})
            if D.latitude.shape[0] > 0:
                self.from_latlon(D.latitude, D.longitude,  filename, 'ATM_Qfit', number=number)
        if file_type in ['ATM_waveform']:
            D=Qfit_data(filename=filename, waveform_format=True)
            if D.latitude.shape[0] > 0:
                self.from_latlon(D.latitude, D.longitude,  filename, 'ATM_waveform', number=number)
        if file_type in ['h5_geoindex']:         
            # read the file as a collection of points
            temp_GI=geo_index().from_file(filename)
            xy_bin=temp_GI.bins_as_array()
            for attr in temp_GI.attrs.keys():
                self.attrs[attr]=temp_GI.attrs[attr]
            self.from_xy(xy_bin, filename, file_type, number=number, fake_offset_val=-1)
        if file_type in ['indexed_h5']:
            self.attrs['SRS_proj4']=SRS_proj4
            h5f=h5py.File(filename,'r')
            xy=[np.array(h5f['INDEX']['bin_x']), np.array(h5f['INDEX']['bin_y'])]
            if 'bin_index' in h5f['INDEX']:
                # this is the type of indexed h5 that has all of the data in single datasets
                i0_i1=h5f['INDEX']['bin_index']
                first_last=[i0_i1[0,:].ravel(), i0_i1[1,:].ravel()]
                fake_offset=None
            else:
                first_last=None
                fake_offset=-1
            self.from_xy(xy, filename, file_type, number=number, first_last=first_last, fake_offset_val=fake_offset)
            h5f.close()
        if file_type in ['indexed_h5_from_matlab']:
            self.attrs['SRS_proj4']=SRS_proj4
            h5f=h5py.File(filename,'r')
            xy=[np.array(h5f['INDEX']['bin_x'])*1000, np.array(h5f['INDEX']['bin_y'])*1000]
            first_last=None
            fake_offset=-1
            self.from_xy(xy, filename, file_type, number=number, first_last=first_last, fake_offset_val=fake_offset)
            h5f.close()

        return self

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
        return self.query_xy([xy_bin[these,0], xy_bin[these,1]], get_data=get_data)

    def intersect(self, other, pad=[0, 0]):
        """
        given a pair of geo indexes return the subsets that are common between the two, optionally padding one or both
        """
        bins_both=set(self.keys()).intersection(other.keys())
        xyB=np.c_[[np.fromstring(key, sep='_') for key in bins_both]]
        if xyB.size==0:
            return None, None
        self_sub=self.copy_subset(xyBin=[xyB[:,0], xyB[:,1]], pad=pad[0])
        other_sub=other.copy_subset(xyBin=[xyB[:,0], xyB[:,1]], pad=pad[1])
        return self_sub, other_sub

    def query_xy(self, xyb, cleanup=True, get_data=True, fields=None, pad=None, dir_root='', strict=False, default_type=None):
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
        
        delta=self.attrs['delta']
        if isinstance(xyb[0], np.ndarray):
            xyb=[xyb[0].copy().ravel(), xyb[1].copy().ravel()]            
        if pad is not None:
            xyb=pad_bins(xyb, pad, delta)

        # make a temporary geo_index to hold the subset of the current geoindex
        # corresponding to xb and yb      
        temp_gi=geo_index(delta=self.attrs['delta'], SRS_proj4=self.attrs['SRS_proj4'])       
        for bin in set(zip(xyb[0], xyb[1])):
           bin_name='%d_%d' % bin
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
                #before or adjacent to the end of the previous, stick them together  
                ii=np.argsort(i0)
                i0=i0[ii]
                i1=i1[ii]
                xy=xy[ii,:]
                keep=np.zeros(len(i0), dtype=bool)
                this=0
                keep[this]=True
                for kk in np.arange(1,len(i0)):
                    if i0[kk]<=i1[this]+1 and i0[kk] > 0 and i0[kk] > 0:
                        keep[kk]=False
                        i1[this]=np.maximum(i1[this], i1[kk])
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
            if strict is True:
                # take the subset of data that rounds exactly to the query (OTW, may get data that extend outside)
                if ~isinstance(query_results, list):
                    query_results=list(query_results)
                for item in query_results:
                    if not hasattr(item,'x'):
                        item.get_xy(self.attrs['SRS_proj4'])
                    keep=np.zeros_like(item.x, dtype=bool)
                    xr=np.round(item.x/delta[0])*delta[0]
                    yr=np.round(item.y/delta[0])*delta[0]
                    for xybi in zip(xyb[0], xyb[1]):
                        ii=(xr==xybi[0]) & (yr==xybi[1])
                        keep[ii]=True
                    item.subset(keep)    
        return query_results
    
    def bins_as_array(self):
        if len(self)>0:
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in self.keys()]]
        else:
            xy_bin=np.c_[[np.fromstring(key, sep='_') for key in self.h5_file_index.keys()]]
        if xy_bin.size > 0:
            return (xy_bin[:,0].ravel(), xy_bin[:,1].ravel())
        else:
            return (np.zeros(0), np.zeros(0))
            
    def bin_latlon(self):
        xy_bin=self.bins_as_array()
        internal_srs=osr.SpatialReference()
        internal_srs.ImportFromProj4(self.attrs['SRS_proj4'])
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation( internal_srs, ll_srs)
        lon, lat, z0 = list(zip(*[ct.TransformPoint(*xy) for xy in zip(np.ravel(xy_bin[:,0]), np.ravel(xy_bin[:,1]), np.ravel(np.zeros_like(xy_bin[:,1])))]))
        return (lat, lon)
        
    def keys_from_xy(self, xy):
        pts=unique_points((xy[0].ravel(), xy[1].ravel()), delta=self.attrs['delta'])
        result=[p1 for p1 in ['%d_%d' % p0 for p0 in zip(pts[0], pts[1])] if p1 in self]
        return result
        
def get_data_for_geo_index(query_results, delta=None, fields=None, data=None, dir_root=''):
    # read the data from a set of query results  
    # Currently the function knows how to read h5_geoindex and ATL06 data.  
    # Append more cases as needed
    if len(dir_root)>0:
        dir_root += '/'
    out_data=list()
    for file_key, result in query_results.items():
        #print(result.__class__)       
        if dir_root is not None:
            this_file = dir_root+file_key
        else:
            this_file=file_key
        if result['type'] == 'h5_geoindex':
            out_data += geo_index().from_file(this_file).query_xy((result['x'], result['y']),  fields=fields, get_data=True)
        if result['type'] == 'ATL06':
            if fields is None:
                fields={None:(u'latitude',u'longitude',u'h_li',u'delta_time')}
            D6_file, pair=this_file.split(':pair')             
            D6=[ATL06_data(beam_pair=int(pair), field_dict=fields).from_file(filename=D6_file, index_range=np.array(temp)) for temp in zip(result['offset_start'], result['offset_end'])]           
            if isinstance(D6,list):
                out_data += D6  
            else:
                data.append(D6)
        if result['type'] == 'ATM_Qfit':
            Qfit_temp=[Qfit_data(filename=this_file, index_range=np.array(temp)) for temp in zip(result['offset_start'], result['offset_end'])]           
            if isinstance(Qfit_temp,list):
                out_data += Qfit_temp
            else:
                out_data.append(Qfit_temp)
        if result['type'] == 'ATM_waveform':
            WF_temp=[Qfit_data(filename=this_file, index_range=np.array(temp), waveform_format=True) for temp in zip(result['offset_start'], result['offset_end'])]
            if isinstance(WF_temp, list):
                out_data += WF_temp
            else:
                out_data.append(WF_temp)
        if result['type'] == 'indexed_h5':
            out_data += [read_indexed_h5_file(this_file, [result['x'], result['y']],  fields=fields, index_range=[result['offset_start'], result['offset_end']])]
        if result['type'] is None:
            out_data=[data.subset(np.arange(temp[0], temp[1])) for temp in zip(result['offset_start'], result['offset_end'])]           
    return out_data

def unique_points(xy, delta=[1, 1]):
    xr=(np.round(np.array(xy[0])/delta[0])*delta[0]).ravel().tolist()
    yr=(np.round(np.array(xy[1])/delta[1])*delta[1]).ravel().tolist()
    xyb=np.concatenate([np.array(xybi).reshape([1,2]) for xybi in set(zip(xr, yr))], axis=0)
    return [xyb[:,0], xyb[:,1]]

def pad_bins(xyb, pad, delta):
    [xp,yp]=np.meshgrid(np.arange(-pad, pad+1)*delta[0], np.arange(-pad, pad+1)*delta[1])
    xp=xp.ravel(); yp=yp.ravel();
    if isinstance(xyb[0],int) or isinstance(xyb[0],float):
        xyb[0]=np.array([xpi+xyb[0] for xpi in xp]) 
        xyb[1]=np.array([ypi+xyb[1] for ypi in yp]) 
    else:
        xyb[0]=np.concatenate([xpi+xyb[0] for xpi in xp]).ravel()
        xyb[1]=np.concatenate([ypi+xyb[1] for ypi in yp]).ravel()

    # keep only the unique members of xb and yb
    xyb = unique_points(xyb, delta)
    return xyb

def read_indexed_h5_file(filename, xy_bin,  fields=['x','y','time'], index_range=[[-1],[-1]]):
    out_data={field:list() for field in fields}
    h5f=h5py.File(filename,'r')
    blank_fields=list()
    if index_range[0][0]>=0:
        # All the geo bins are together.  Use the index_range variable to read
        for field in fields:
            for i0_i1 in zip(index_range):
                if field in h5f:
                    out_data[field].append([h5f[field][i0_i1[0]:i0_i1[1]]])
                else:
                    blank_fields.append(field)
    else:
        # this is a file with distinct bins, each with its own set of datasets
        for xy in zip(xy_bin):
            bin_name='%dE_%dN' % xy
            for field in fields:
                if field in h5f[bin_name]:
                    out_data[field].append([h5f[bin_name][field]])
                else:
                    blank_fields.append(field)
    h5f.close()
    return point_data().from_list(out_data, list_of_fields=fields)

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

def index_list_for_files(filename_list, file_type, delta, SRS_proj4, dir_root=''):
    index_list=list()
    for filename in filename_list:
        index_list.append(geo_index(SRS_proj4=SRS_proj4).for_file(filename, file_type, delta, SRS_proj4, dir_root=dir_root, number=0))
    return index_list