# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:46:21 2017

Class to read and manipulate ATL06 data.  Currently set up for Ben-style fake data, should be modified to work with the official ATL06 prodct foramt

@author: ben
"""
import h5py
import numpy as np
from ATL06_pair import ATL06_pair
import matplotlib.pyplot as plt 
import re

class ATL06_data:
    np.seterr(invalid='ignore')
    def __init__(self, filename=None, beam_pair=1, x_bounds=None, y_bounds=None, index_range=None, field_dict=None, list_of_fields=None, list_of_data=None, from_dict=None, NICK=None): 
        if field_dict is None:
            if NICK is not None:
                field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                            'ground_track':['cycle','x_atc', 'y_atc','seg_azimuth','sigma_geo_at','sigma_geo_xt'],
                            'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','dh_fit_dy','h_rms_misft','h_robust_spread','signal_selection_source','snr_significance'],
                            'geophysical':['bsnow_conf','bsnow_h','cloud_flg_asr','cloud_flg_atm','r_eff','tide_ocean']} 
            else:
                field_dict={'land_ice_height':['delta_time','dh_fit_dx','dh_fit_dy','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id'], 
                            'ground_track':['cycle','x_atc', 'y_atc','seg_azimuth'],
                            'fit_statistics':['dh_fit_dx_sigma','h_robust_spread','snr_significance','signal_selection_source']} 

        if list_of_fields is None:
            list_of_fields=list()
            for group in field_dict.keys():
                for field in field_dict[group]:
                    list_of_fields.append(field)

        self.list_of_fields=list_of_fields
        if list_of_data is not None:
            self.build_from_list_of_data(list_of_data)
            return None     
        if from_dict is not None:
            self.list_of_fields=list_of_fields
            for field in list_of_fields:
                setattr(self, field, from_dict[field])
            return
        # read from a file if specified
        if filename is not None:
            # read a list of files if list provided
            if isinstance(filename, (list, tuple)):
                D6_list=[ATL06_data(filename=thisfile, field_dict=field_dict, beam_pair=beam_pair, index_range=index_range, x_bounds=x_bounds, y_bounds=y_bounds, NICK=NICK) for thisfile in filename]
                self.build_from_list_of_data(D6_list)
            elif isinstance(filename, (str)):
                # this happens when the input filename is a string, not a list
                self.read_from_file(filename, field_dict, beam_pair=beam_pair, index_range=index_range, x_bounds=x_bounds, y_bounds=y_bounds, NICK=NICK)
            else:
                raise TypeError
        else:
            # no file specified, set blank
            for field in list_of_fields:
                setattr(self, field, np.zeros((2,0)))       
          
    def read_from_file(self, filename, field_dict, index_range=None, x_bounds=None, y_bounds=None, beam_pair=None, NICK=None): 
        seconds_per_day=24*3600.
        h5_f=h5py.File(filename,'r')        
        beam_names=['gt%d%s' %(beam_pair, b) for b in ['l','r']]
        if beam_names[0] not in h5_f or beam_names[1] not in h5_f:        
            # return empty data structure
            for group in field_dict:
                for field in group:
                    setattr(self, field, np.zeros((0,2)))
            return
        # find cycle number in filename
        m=re.search(r"TrackData_(.*?)/Track_(.*?).h5",filename)
        cycle_number=np.int(m.group(1))
        #N.B. with real ATL06 data we will not want to apply the reference time offset to the delta time.
        time_offset=0.
        if NICK:
            time_offset=h5_f.attrs['reference_time']*seconds_per_day
        #print('line 61,',NICK,filename)
        if beam_names[0] not in h5_f.keys():
            return None
        if index_range is None:
            index_range=[0, h5_f[beam_names[0]]['land_ice_segments']['h_li'].size]
        
        for group in field_dict.keys():
            for field in field_dict[group]:
                if field not in self.list_of_fields:
                    self.list_of_fields.append(field)
                try:
                    if NICK is not None:
                        if group is None:
                            setattr(self, field, np.c_[
                                np.array(h5_f[beam_names[0]]['land_ice_segments'][field][0,index_range[0]:index_range[1]]).transpose(),  
                                np.array(h5_f[beam_names[1]]['land_ice_segments'][field][0,index_range[0]:index_range[1]]).transpose()])
                        else:
                            if 'ground' in group and 'cycle' in field:  # currently groundtrack/cycle NaN
                                n_vals=np.array(h5_f[beam_names[0]]['land_ice_segments'][group][field][0,index_range[0]:index_range[1]]).size
                                setattr(self, field, np.zeros( [n_vals, 2])+cycle_number)
                            else:
                                setattr(self, field, np.c_[
                                np.array(h5_f[beam_names[0]]['land_ice_segments'][group][field][0,index_range[0]:index_range[1]]).transpose(),  
                                np.array(h5_f[beam_names[1]]['land_ice_segments'][group][field][0,index_range[0]:index_range[1]]).transpose()])
                    else:
                        setattr(self, field, np.c_[
                        np.array(h5_f[beam_names[0]][group][field][0,index_range[0]:index_range[1]]).transpose(),  
                        np.array(h5_f[beam_names[1]][group][field][0,index_range[0]:index_range[1]]).transpose()])
                            
                except KeyError:
                    print("could not read %s/%s" % (group, field))
                    setattr(self, field, np.zeros_like(self.delta_time)+np.NaN)
        if NICK is not None:
            self.sigma_geo_h =np.zeros_like(self.h_li)+0.03
            self.sigma_geo_xt=np.zeros_like(self.h_li)+6.5
            self.sigma_geo_at=np.zeros_like(self.h_li)+6.5
            self.delta_time=self.delta_time*24.*3600.+time_offset
        return

    def append(self, D):
        for field in self.list_of_fields:
            setattr(self, np.c_[getattr(self, field), getattr(D, field)])
        return        

    def build_from_list_of_data(self, D6_list):
        try:
            for field in self.list_of_fields:
                data_list=[getattr(this_D6, field) for this_D6 in D6_list]       
                setattr(self, field, np.concatenate(data_list, 0))
        except TypeError:
            for field in self.list_of_fields:
                setattr(self, field, getattr(D6_list, field))
        return 
    
    def index(self, index):
        for field in self.list_of_fields:
            setattr(self, field, getattr(self, field)[index,:])
        return
        
    def subset(self, index, by_row=True, datasets=None):
        dd=dict()
        if datasets is None:
            datasets=self.list_of_fields
        for field in datasets:
            if by_row is not None and by_row:
                dd[field]=getattr(self, field)[index,:]
            else:
                dd[field]=getattr(self, field)[index,:].ravel()[index]
        return ATL06_data(from_dict=dd, list_of_fields=datasets)
            
    def copy(self):
        return ATL06_data(list_of_data=(self), list_of_fields=self.list_of_fields)
    
    def plot(self, valid_pairs=None, valid_segs=None):
        colors=('r','b')
        for col in (0, 1):
            plt.errorbar(self.x_atc[:,col], self.h_li[:,col], yerr=self.h_li_sigma[:, col], c=colors[col], marker='.', linestyle='None', markersize=4);
        if valid_segs is not None:
            for col in (0, 1):
                plt.plot(self.x_atc[valid_segs[col], col], self.h_li[valid_segs[col], col],'marker','x',c=colors[col])

        if valid_pairs is not None:
            for col in (0, 1):
                plt.plot(self.x_atc[valid_pairs, col], self.h_li[valid_pairs, col],'marker','o',c=colors[col])
        plt.ylim(np.amin(self.h_li[self.atl06_quality_summary[:,1]==0,1])-5., np.amax(self.h_li[self.atl06_quality_summary[:,1]==0 ,1])+5 ) 
        #plt.show()
        return

    def get_pairs(self, datasets=None):
        pair_list=list()
        for i in np.arange(self.h_li.shape[0]):
            pair_list.append(ATL06_pair(D6=self.subset(i, by_row=True, datasets=datasets)))
        all_pairs=ATL06_pair(pair_data=pair_list)
        return all_pairs

 