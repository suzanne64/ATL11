# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:46:21 2017

Class to read and manipulate ATL06 data.  Currently set up for Ben-style fake data, should be modified to work with the official ATL06 prodct foramt

@author: ben
"""
import h5py
import numpy as np
from ATL11.ATL06_pair import ATL06_pair
from osgeo import osr
import matplotlib.pyplot as plt 

class ATL06_data:
    np.seterr(invalid='ignore')
    def __init__(self, beam_pair=2, field_dict=None, list_of_fields=None, NICK=None): 
        """
        Initialize an ATl06 data structure. 
        
        ATL06_data has attributes:
            list_of_fields: list of all the fields defined for the current structure
            field_dict: dictionary that shows how to read data from the file.
            beam_pair:  The ICESat-2 beam pair from which the data come
            size: the number of elements in each of its data fields
            shape: the shape of each of its data fields
        It has a variable number of data fields that are assigned during reading, 
        each of which contains an  N x 2 numpy array of doubles that holds
        data from the left and right beams of the ATL06 pair (gtxl and gtxr)
        """
        if field_dict is None:
            self.field_dict=self.__default_field_dict__()
        else:
            self.field_dict=field_dict
            
        if list_of_fields is None:
            list_of_fields=list()
            for group in self.field_dict.keys():
                for field in self.field_dict[group]:
                    list_of_fields.append(field)
                    
        self.beam_pair=beam_pair
        self.beam_type=['weak','strong'] # defaults for the early part of the mission
        self.orbit=np.NaN
        self.rgt=np.NaN
        self.file=None
        self.list_of_fields=list_of_fields
        for field in list_of_fields:
            setattr(self, field, np.zeros((0,2)))   
        self.size=0
        self.shape=(0,2)
        
    def __update_size_and_shape__(self):
        """
        When data size and shape may have changed, update the size and shape atttributes
        """
        temp=getattr(self, self.list_of_fields[0])
        self.size=temp.size
        self.shape=temp.shape
        return self
        
    def __getitem__(self, rows):
        """
        Experimental: index the whole structure at once using square brackets
        """
        return self.copy().subset(rows)
        
    def __default_field_dict__(self):
        """
        Define the default fields that get read from the h5 file
        """
        field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                    'ground_track':['x_atc', 'y_atc','seg_azimuth','sigma_geo_at','sigma_geo_xt'],
                    'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','h_mean', 'dh_fit_dy','h_rms_misfit','h_robust_sprd','n_fit_photons', 'signal_selection_source','snr_significance','w_surface_window_final'],
                    'geophysical':['bsnow_conf','bsnow_h','cloud_flg_asr','cloud_flg_atm','r_eff','tide_ocean'],
                    'derived':['valid']}         
        return field_dict
        
    def from_file(self, filename, index_range=None, x_bounds=None, y_bounds=None): 
        """
        Read data from a file.
        """
        self.file=filename
        h5_f=h5py.File(filename,'r')
        
        beam_names=['gt%d%s' %(self.beam_pair, b) for b in ['l','r']]
        if beam_names[0] not in h5_f or beam_names[1] not in h5_f:        
            # return empty data structure
            for group in self.field_dict:
                for field in self.field_dict[group]:
                    setattr(self, field, np.zeros((0,2)))
                self.__update_size_and_shape__()
            return self
        if beam_names[0] not in h5_f.keys():
            return None
        # get the strong/weak beam info
        for count, beam_name in enumerate(beam_names):
            try:
                self.beam_type[count]=h5_f[beam_name]['atlas_beam_type']
            except KeyError:
                pass  # leave the beam type as the default
        # read the orbit number
        try:
            self.rgt=int(h5_f['orbit_info']['rgt'][0])
            self.orbit=int(h5_f['orbit_info']['orbit_number'][0])
        except:
            pass
        if index_range is None or index_range[1]==-1:
            index_range=[0, h5_f[beam_names[0]]['land_ice_segments']['h_li'].size]
        n_vals=index_range[-1]-index_range[0]
        
        for group in self.field_dict.keys():
            for field in self.field_dict[group]:
                if field not in self.list_of_fields:
                    self.list_of_fields.append(field)
                try:
                    #bad_val=-9999 # default value, works for most groups
                    if group is None:
                        #The None group corresponds to the top level of the land_ice_segments hirearchy
                        # most datasets have a __fillValue attribute marking good data, but some don't
                        try:
                            bad_val=h5_f[beam_names[0]]['land_ice_segments'][field].attrs['_FillValue']
                        except KeyError:
                            #latitude and longitude don't have fill values, but -9999 works OK.
                            bad_val=-9999
                        data=np.c_[
                            np.array(h5_f[beam_names[0]]['land_ice_segments'][field][index_range[0]:index_range[1]]).transpose(),  
                            np.array(h5_f[beam_names[1]]['land_ice_segments'][field][index_range[0]:index_range[1]]).transpose()]
                    elif group is "orbit_info":
                        data=np.zeros((n_vals, 2))+h5_f['orbit_info'][field]
                        bad_val=-9999
                    elif group is "derived" and field is "valid":
                        data=np.ones((index_range[1]-index_range[0], 2), dtype='bool')
                        bad_val=0
                    else:
                        
                        # All other groups are under the land_ice_segments/group hirearchy
                         try:
                            bad_val=h5_f[beam_names[0]]['land_ice_segments'][group][field].attrs['_FillValue']
                         except KeyError:
                            #flags don't have fill values, but -9999 works OK.
                            bad_val=-9999
                         try:
                             data=np.c_[
                                np.array(h5_f[beam_names[0]]['land_ice_segments'][group][field][index_range[0]:index_range[1]]).transpose(),  
                                np.array(h5_f[beam_names[1]]['land_ice_segments'][group][field][index_range[0]:index_range[1]]).transpose()]
                         except KeyError:
                             print("missing hdf field for land_ice_segments/%s/%s" % (group, field))
                             data=np.zeros((index_range[1]+1-index_range[0], 2))+bad_val
                    # identify the bad data elements before converting the field to double
                    bad=data==bad_val
                    if field is not 'valid':
                        data=data.astype(np.float64)
                    # mark data that are invalid (according to the h5 file) with NaNs
                    data[bad]=np.NaN
                    setattr(self, field, data)                            
                except KeyError:
                    print("could not read %s/%s" % (group, field))
                    setattr(self, field, np.zeros( [n_vals, 2])+np.NaN)
        if 'atl06_quality_summary' in self.list_of_fields:
            # add a non-broken version of the atl06 quality summary                    
            setattr(self, 'atl06_quality_summary', (self.h_li_sigma > 1) | (self.h_robust_sprd > 1) | (self.snr_significance > 0.02))
        h5_f.close()
        self.__update_size_and_shape__()
        # assign fields that must be copied from single-value attributes in the 
        # h5 file
        if 'cycle_number' in self.list_of_fields:
            # get the cycle number
            try:
                cycle_number=h5_f['ancillary_data']['start_cycle']  
            except KeyError:
                cycle_number=-9999
            setattr(self, 'cycle_number', cycle_number+np.zeros(self.shape, dtype=np.float64))
        return self
    
    def get_xy(self, proj4_string, EPSG=None):
        # method to get projected coordinates for the data.  Adds 'x' and 'y' fields to the data, optionally returns 'self'
        out_srs=osr.SpatialReference()
        if proj4_string is None and EPSG is not None:
            out_srs.ImportFromProj4(EPSG)
        else:
            out_srs.ImportFromProj4(proj4_string)
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        if self.latitude.size==0:
            self.x=np.zeros_like(self.latitude)
            self.y=np.zeros_like(self.latitude)
        else:
            x, y, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(self.longitude), np.ravel(self.latitude), np.zeros_like(np.ravel(self.latitude)))]))
            self.x=np.reshape(x, self.latitude.shape)
            self.y=np.reshape(y, self.longitude.shape)
        if 'x' not in self.list_of_fields:
            self.list_of_fields += ['x','y']
        return self
    
    def append(self, D):
        """
        Append the fields of two ATL06_data instances
        """
        for field in self.list_of_fields:
            setattr(self, np.c_[getattr(self, field), getattr(D, field)])
        self.__update_size_and_shape__()    
        return self

    def from_list(self, D6_list):
        """
        Append the fields of several ATL06_data instances.
        """
        try:
            for field in self.list_of_fields:
                data_list=[getattr(this_D6, field) for this_D6 in D6_list]       
                setattr(self, field, np.concatenate(data_list, 0))
        except TypeError:
            for field in self.list_of_fields:
                setattr(self, field, getattr(D6_list, field))
        self.__update_size_and_shape__()
        return self
    
    def from_dict(self, D6_dict):
        """
        Build an ATL06_data from a dictionary
        """
        for field in self.list_of_fields:
            try:
                setattr(self, field, D6_dict[field])
            except KeyError:
                setattr(self, field, None)
        self.__update_size_and_shape__()
        return self
    
    def index(self, rows):
        """
        Select a subset of rows within a given ATL06_data instance (modify in place)
        """
        for field in self.list_of_fields:
            setattr(self, field, getattr(self, field)[rows,:])
        self.__update_size_and_shape__()
        return self
        
    def subset(self, index, by_row=True, datasets=None):
        """
        Make a subsetted copy of the current ATL06_data instance
        """
        dd=dict()
        if datasets is None:
            datasets=self.list_of_fields
        for field in datasets:
            if by_row is not None and by_row:
                dd[field]=getattr(self, field)[index,:]
            else:
                dd[field]=getattr(self, field)[index,:].ravel()[index]
        return ATL06_data(list_of_fields=datasets, beam_pair=self.beam_pair).from_dict(dd)
    
    def copy(self):
        return ATL06_data( list_of_fields=self.list_of_fields, beam_pair=self.beam_pair).from_list((self))
    
    def plot(self, valid_pairs=None, valid_segs=None):
        colors=('r','b')
        for col in (0, 1):
            #plt.errorbar(self.x_atc[:,col], self.h_li[:,col], yerr=self.h_li_sigma[:, col], c=colors[col], marker='.', linestyle='None', markersize=4);
            plt.plot(self.x_atc[:,col], self.h_li[:,col], c=colors[col], marker='.', linestyle='None', markersize=2);
        if valid_segs is not None:
            for col in (0, 1):
                plt.plot(self.x_atc[valid_segs[col], col], self.h_li[valid_segs[col], col], marker='x',c=colors[col])

        if valid_pairs is not None:
            for col in (0, 1):
                plt.plot(self.x_atc[valid_pairs, col], self.h_li[valid_pairs, col], marker='o',c=colors[col])
        plt.ylim(np.nanmin(self.h_li[self.atl06_quality_summary[:,1]==0,1])-5., np.nanmax(self.h_li[self.atl06_quality_summary[:,1]==0 ,1])+5 ) 
        #plt.show()
        return

    def get_pairs(self, datasets=None):
        pair_list=list()
        for i in np.arange(self.h_li.shape[0]):
            pair_list.append(ATL06_pair(D6=self.subset(i, by_row=True, datasets=datasets)))
        all_pairs=ATL06_pair(pair_data=pair_list)
        return all_pairs

 