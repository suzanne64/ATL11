# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt 
import h5py, re, os, csv 
from ATL11_misc import ATL11_defaults


class ATL11_group(object):
    # Class to contain an ATL11 structure
    # in ATL11 groups, some datasets have one value per reference point (per_pt_fields)
    # some have one value for each reference point and each cycle (full_fields)
    # some have one value for each point and each polynomial coefficient (poly_fields)
    # all of these are initialized to arrays of the appropriate size, filled with NaNs

    def __init__(self, N_pts, N_cycles, N_coeffs, per_pt_fields=None, full_fields=None, poly_fields=None):
        # input variables:
        #  N_pts: Number of reference points to allocate
        #  N_cycles: Number of cycles of data to allocate
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
                setattr(self, field, np.nan + np.zeros([N_pts, N_cycles]))
        if poly_fields is not None:
            for field in poly_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, N_coeffs]))
        # assemble the field names into lists:
        self.per_pt_fields=per_pt_fields
        self.full_fields=full_fields
        self.poly_fields=poly_fields
        self.list_of_fields=self.per_pt_fields+self.full_fields+self.poly_fields
        
                
class valid_mask:
    # class to hold validity flags for different attributes
    def __init__(self, dims, fields):
        for field in fields:
            setattr(self, field, np.zeros(dims, dtype='bool'))

class ATL11_data(object):
    # class to hold ATL11 data in ATL11_groups
    def __init__(self, N_pts=1, N_cycles=1, N_coeffs=9, from_file=None):
        self.Data=[]
        self.DOPLOT=None
 
        # define empty records here based on ATL11 ATBD
        # read in parameters information in .csv
#        with open('ATL11_output_attrs_reps.csv','r') as attrfile:
#            reader=csv.DictReader(attrfile)  # must save excel file with format: Comma Separated Values (.csv) 
#            # get group names
#            grp_names=list(set([row['group'] for row in reader]))
#            for item in grp_names:
#                print(item)
#                #print(row['field'] for item in row['group'] in reader)
            
        # Table 4-1
        self.corrected_h=ATL11_group(N_pts, N_cycles, N_coeffs, per_pt_fields=['ref_pt_lat','ref_pt_lon','ref_pt_number'], 
                                       full_fields=['mean_cycle_time','cycle_h_shapecorr','cycle_h_shapecorr_sigma','cycle_h_shapecorr_sigma_systematic','quality_summary'],
                                       poly_fields=[])   
        # Table 4-2        
        self.ref_surf=ATL11_group(N_pts, N_cycles, N_coeffs, per_pt_fields=['complex_surface_flag','fit_curvature','fit_E_slope','fit_N_slope','n_deg_x','n_deg_y',
                                                                              'N_cycle_avail','N_cycle_used','ref_pt_number','ref_pt_x_atc','ref_pt_y_atc','rgt_azimuth',
                                                                              'slope_change_rate_x','slope_change_rate_y','slope_change_rate_x_sigma','slope_change_rate_y_sigma',
                                                                              'surf_fit_misfit_chi2r','surf_fit_misfit_RMS','surf_fit_quality_summary'],
                                    full_fields=[], poly_fields=['poly_coeffs','poly_coeffs_sigma'])
        # Table 4-3
        self.cycle_stats=ATL11_group(N_pts, N_cycles, N_coeffs, per_pt_fields=['ref_pt_number'],
                                      full_fields=['ATL06_summary_zero_count','h_robust_spread_mean','h_rms_misft_mean','r_eff_mean','tide_ocean_mean',
                                                   'cloud_flg_atm_best','cloud_flg_asr_best','bsnow_h_mean','bsnow_conf_best',
                                                   'x_atc_mean','y_atc_mean','cycle_included_in_fit','cycle_seg_count','strong_beam_number',
                                                   'latitude_mean','longitude_mean','min_signal_selection_source','min_SNR_significance',
                                                   'sigma_geo_h_mean','sigma_geo_at_mean','sigma_geo_xt_mean','h_uncorr_mean'], poly_fields=[])
        # Table 4-4, not yet implemented
        #self.crossing_track_data=ATL11_group(N_pts, N_cycles, N_coeffs, per_pt_fields=[],
        self.slope_change_t0=None

    def all_fields(self):
        # return a list of all the fields in an ATL11 instance
        all_vars=[]
        di=vars(self)  # a dictionary
        for item in di.keys():
            if hasattr(getattr(self,item),'list_of_fields'):
                all_vars.append(getattr(getattr(self,item),'list_of_fields'))
        all_vars=[y for x in all_vars for y in x] # flatten list of lists
        return all_vars
        
    def from_list(self, P11_list):  
        # Assemble an ATL11 data instance from a list of ATL11 points.  
        # Input: list of ATL11 point instances
        # loop over variables in ATL11_data (self)
        self.__init__(N_pts=len(P11_list), N_cycles=P11_list[0].corrected_h.cycle_h_shapecorr.shape[1], N_coeffs=P11_list[0].ref_surf.poly_coeffs.shape[1])
    
        for group in vars(self).keys():
            # check if each variable is an ATl11 group
            if  not isinstance(getattr(self,group), ATL11_group):
                continue
            for field in getattr(self, group).per_pt_fields:
                temp=np.ndarray(shape=[len(P11_list),],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        if 'ref_pt_number' in field:
                            temp[ii]=P11.ref_pt_number
                        else:
                            temp[ii]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)
                        
            for field in getattr(self,group).full_fields:
                temp=np.ndarray(shape=[len(P11_list),P11_list[0].N_cycles],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        temp[ii,:]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)
                        
            for field in getattr(self,group).poly_fields:
                temp=np.ndarray(shape=[len(P11_list),P11_list[0].N_coeffs],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        temp[ii,:]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)
        self.slope_change_t0=P11_list[0].slope_change_t0                                    
        return self
    
    def from_file(self, filename=None):
      
        FH=h5py.File(filename,'r')
        N_pts=FH['corrected_h']['cycle_h_shapecorr'].shape[0]
        N_cycles=FH['corrected_h']['cycle_h_shapecorr'].shape[1]
        N_coeffs=FH['ref_surf']['poly_coeffs'].shape[1]
        self.__init__(N_pts=N_pts, N_cycles=N_cycles, N_coeffs=N_coeffs)
        for group in ('corrected_h','ref_surf','cycle_stats'):
            for field in FH[group].keys():
                setattr(getattr(self, group), field, np.array(FH[group][field]))
        FH=None
        return self
        
    def write_to_file(self, fileout, params_11=None):
        # Generic code to write data from an ATL11 object to an h5 file
        # Input:
        #   fileout: filename of hdf5 filename to write
        # Optional input:
        #   parms_11: ATL11_defaults structure
        if os.path.isfile(fileout):
            os.remove(fileout)
        f = h5py.File(fileout,'w')

        # This code is based on the filename structure used in our demonstration code.
        m=re.search(r"Pair(.*?).h5",fileout)
        f.attrs['pairTrack']=m.group(1)
        m=re.search(r"Track(.*?)_Pair",fileout)
        f.attrs['ReferenceGroundTrack']=m.group(1)
        # put default parameters as top level attributes
        if params_11 is None:
            params_11=ATL11_defaults()
        # write each variable in params_11 as an attribute
        for param in  vars(params_11).keys():
            try:
                f.attrs[param]=getattr(params_11, param)
            except:  
                print("write_to_file:could not automatically set parameter: %s" % param)
                
        # put attrubutes into dict, where parameters from Tables 4- in ATBD are keys      
        with open('ATL11_output_attrs.csv','r') as attrfile:
            reader=csv.DictReader(attrfile)  # must save excel file with format: Comma Separated Values (.csv) 
            attrs_list=[x for x in reader.fieldnames if x != 'field' and x != 'group']
            field_attrs={row['field']: {attrs_list[ii]:row[attrs_list[ii]] for ii in range(len(attrs_list))}  for row in reader}

        # write data to file            
        for group in vars(self).keys():
            if hasattr(getattr(self,group),'list_of_fields'):
                grp = f.create_group(group)
                if 'ref_surf' in group:
                    grp.attrs['poly_exponent_x']=np.array([item[0] for item in params_11.poly_exponent_list], dtype=int)
                    grp.attrs['poly_exponent_y']=np.array([item[1] for item in params_11.poly_exponent_list], dtype=int) 
                    grp.attrs['slope_change_t0'] =np.mean(self.slope_change_t0).astype('int')
                list_vars=getattr(self,group).list_of_fields
                if list_vars is not None:
                    for field in list_vars: 
                        dset = grp.create_dataset(field,data=getattr(getattr(self,group),field))
                        for attr in attrs_list:
                            dset.attrs[attr] = field_attrs[field][attr]
        f.close()    
        return
        
    def plot(self):
        # method to plot the results.  At present, this plots corrected h AFN of x_atc
        n_cycles=self.corrected_h.cycle_h_shapecorr.shape[1]
        HR=np.nan+np.zeros((n_cycles, 2))
        h=list()
        #plt.figure(1);plt.clf()
        for cycle in range(n_cycles):
            xx=self.ref_surf.ref_pt_x_atc
            zz=self.corrected_h.cycle_h_shapecorr[:,cycle]
            ss=self.corrected_h.cycle_h_shapecorr_sigma[:,cycle]
            good=np.abs(ss)<50   
            if np.any(good):               
                h0=plt.errorbar(xx[good],zz[good],ss[good], marker='o',picker=5)
                h.append(h0)
                HR[cycle,:]=np.array([zz[good].min(), zz[good].max()])
                #plt.plot(xx[good], zz[good], 'k',picker=None)
        temp=self.corrected_h.cycle_h_shapecorr.copy()
        temp[self.corrected_h.cycle_h_shapecorr_sigma>20]=np.nan
        temp=np.nanmean(temp, axis=1)
        plt.plot(xx, temp, 'k.', picker=5)
        plt.ylim((np.nanmin(HR[:,0]),  np.nanmax(HR[:,1])))
        plt.xlim((np.nanmin(xx),  np.nanmax(xx)))
        return h
