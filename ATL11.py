# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
from poly_ref_surf import poly_ref_surf
import matplotlib.pyplot as plt 
from RDE import RDE
import scipy.sparse as sparse
from scipy import linalg 
from scipy import stats
import time, h5py, re, os
 
#import collections

class ATL11_group:
    # this is a data group for an ATL11 structure        
    # in ATL11 groups, some datasets have one value per reference point (per_pt_fields)
    # some have one value for each refrence point and each cycle (full_fields)
    # some have one value for each point and each polynomial coefficient (poly_fields)
    # all of these are initialized to arrays of the appropriate size, filled with NaNs

    def __init__(self, N_ref_pts, N_reps, N_coeffs, per_pt_fields=None, full_fields=None, poly_fields=None):
        # input variables:
        #  N_ref_pts: Number of reference points to allocate
        #  N_reps: Number of cycles of data to allocate
        #  N_coeffs: Number of polynomial coefficients to allocate
        #  per_pt_fields: list of fields that have one value per reference point
        #  full_fields: list of fields that have one value per cycle per reference point
        #  poly_fields: list of fields that have one value per polynomial degree combination per reference point        
        
        # assign fields of each type to their appropriate shape and size
        if per_pt_fields is not None:
            for field in per_pt_fields:
                setattr(self, field, np.nan + np.zeros([N_ref_pts, 1]))
        if full_fields is not None:
            for field in full_fields:
                setattr(self, field, np.nan + np.zeros([N_ref_pts, N_reps]))
        if poly_fields is not None:
            for field in poly_fields:
                setattr(self, field, np.nan + np.zeros([N_ref_pts, N_coeffs]))
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

class ATL11_data:
    # class to hold ATL11 data in ATL11_groups
    def __init__(self, N_ref_pts, N_reps, N_coeffs=9):
        self.Data=[]
        self.DOPLOT=None
        # define empty records here based on ATL11 ATBD
        # Table 4-1
        self.corrected_h=ATL11_group(N_ref_pts, N_reps, N_coeffs, per_pt_fields=['ref_pt_lat','ref_pt_lon','ref_pt_number'], 
                                       full_fields=['mean_pass_time','pass_h_shapecorr','pass_h_shapecorr_sigma','pass_h_shapecorr_sigma_systematic','quality_summary'],
                                       poly_fields=[])   
        # Table 4-2        
        self.ref_surf=ATL11_group(N_ref_pts, N_reps, N_coeffs, per_pt_fields=['complex_surface_flag','fit_curvature','fit_E_slope','fit_N_slope','fit_y_slope','n_deg_x','n_deg_y',
                                                                              'N_pass_avail','N_pass_used','ref_pt_number','ref_pt_x_atc','ref_pt_y_atc','rgt_azimuth',
                                                                              'slope_change_rate_x','slope_change_rate_y','slope_change_rate_x_sigma','slope_change_rate_y_sigma',
                                                                              'surf_fit_misfit_chi2','surf_fit_misfit_RMS','surf_fit_quality_summary'],
                                    full_fields=[], poly_fields=['poly_coeffs','poly_coeffs_sigma'])
        # Table 4-3
        self.pass_stats=ATL11_group(N_ref_pts, N_reps, N_coeffs, per_pt_fields=['ref_pt_number'],
                                      full_fields=['ATL06_summary_zero_count','h_robust_spread_mean','h_li_rms_mean','r_eff_mean','tide_ocean_mean',
                                                   'cloud_flg_atm_best','cloud_flg_asr_best','bsnow_h_mean','bsnow_conf_best',
                                                   'y_atc_mean','x_atc_mean','pass_included_in_fit','pass_seg_count','strong_beam_number',
                                                   'mean_pass_lat','mean_pass_lon','min_signal_selection_source','min_SNR_significance',
                                                   'sigma_geo_h_mean','sigma_geo_at_mean','sigma_geo_xt_mean','h_uncorr_mean'], poly_fields=[])
        # Table 4-4
        #self.crossing_track_data=ATL11_group(N_ref_pts, N_reps, N_coeffs, per_pt_fields=[],
                                 
        # this group is used to hold output data attributes                             
        #self.non_product=ATL11_group(N_ref_pts, N_reps, N_coeffs, per_pt_fields=['slope_change_t0'], full_fields=[], poly_fields=['poly_exponent_x','poly_exponent_y'])
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
        di=vars(self)  # a dictionary
        for group in di.keys():
            # want the attribute 'list_of_fields' of ATL11_data attribute ATL11_group
            if hasattr(getattr(self,group),'list_of_fields'):
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
                    temp=np.ndarray(shape=[len(P11_list),P11_list[0].N_reps],dtype=float)
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
        
    def write_to_file(self, fileout, params_11=None):
        if os.path.isfile(fileout):
            os.remove(fileout)
        # Generic code to write data from an object to an h5 file 
        f = h5py.File(fileout,'w')
        # set file attributes
        m=re.search(r"Pair(.*?).h5",fileout)
        f.attrs['pairTrack']=m.group(1)
        m=re.search(r"Track(.*?)_Pair",fileout)
        f.attrs['ReferenceGroundTrack']=m.group(1)
        # put default parameters as top level attributes
        if params_11 is None:
            params_11=ATL11_defaults()
        params_di=vars(params_11)
        for param in  params_di.keys():
            try:
                f.attrs[param]=getattr(params_11, param)
            except: 
                print("write_to_file:could no automatically set parameter: %s" % param)

        # write data to file            
        di=vars(self)  # a dictionary
        for group in di.keys():
            if hasattr(getattr(self,group),'list_of_fields'):
                grp = f.create_group(group)
                if 'ref_surf' in group:
                    grp.attrs['poly_exponent_x']=np.array([item[0] for item in params_11.poly_exponent_list], dtype=int)
                    grp.attrs['poly_exponent_y']=np.array([item[0] for item in params_11.poly_exponent_list], dtype=int) 
                    grp.attrs['slope_change_t0'] =np.mean(self.slope_change_t0).astype('int')
                list_vars=getattr(self,group).list_of_fields
                if list_vars is not None:
                    for field in list_vars: 
                        grp.create_dataset(field,data=getattr(getattr(self,group),field))
        f.close()    
        return
        
    def plot(self):
        n_cycles=self.corrected_h.pass_h_shapecorr.shape[1]
        HR=np.nan+np.zeros((n_cycles, 2))
        h=list()
        #plt.figure(1);plt.clf()
        for cycle in range(n_cycles):
            xx=self.ref_surf.ref_pt_x_atc
            zz=self.corrected_h.pass_h_shapecorr[:,cycle]
            ss=self.corrected_h.pass_h_shapecorr_sigma[:,cycle]
            good=np.abs(ss)<50   
            if np.any(good):               
                h0=plt.errorbar(xx[good],zz[good],ss[good], marker='o',picker=5)
                h.append(h0)
                HR[cycle,:]=np.array([zz[good].min(), zz[good].max()])
                #plt.plot(xx[good], zz[good], 'k',picker=None)
        temp=self.corrected_h.pass_h_shapecorr.copy()
        temp[self.corrected_h.pass_h_shapecorr_sigma>20]=np.nan
        temp=np.nanmean(temp, axis=1)
        plt.plot(xx, temp, 'k.', picker=5)
        plt.ylim((np.nanmin(HR[:,0]),  np.nanmax(HR[:,1])))
        plt.xlim((np.nanmin(xx),  np.nanmax(xx)))
        return h
        
class ATL11_point:
    def __init__(self, N_pairs=1, x_atc_ctr=np.NaN,  y_atc_ctr=np.NaN, track_azimuth=np.NaN, max_poly_degree=[1, 1], N_reps=12, params_11=None, slope_change_t0=None):
        if params_11 is None:        
            self.params_11=ATL11_defaults()
        else:
            self.params_11=params_11
        self.N_pairs=N_pairs
        self.N_reps=N_reps     
        self.N_coeffs=self.params_11.N_coeffs
        self.x_atc_ctr=x_atc_ctr
        self.y_atc_ctr=y_atc_ctr
        self.track_azimuth=track_azimuth
        self.z_poly_fit=None
        self.mx_poly_fit=None
        self.my_poly_fit=None
        self.ref_surf_slope_x=np.NaN
        self.ref_surf_slope_y=np.NaN
        self.slope_change_t0=slope_change_t0
        self.valid_segs =valid_mask((N_pairs,2),  ('data','x_slope','y_slope' ))  #  2 cols, boolan, all F to start
        self.valid_pairs=valid_mask((N_pairs,1), ('data','x_slope','y_slope', 'all','ysearch'))  # 1 col, boolean
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.status=dict()
        self.DOPLOT=None
        # copy ATL11_groups to P11
        temp_D11=ATL11_data(1, N_reps, N_coeffs=self.N_coeffs)
        di=vars(temp_D11)
        for item in di.keys():
            if hasattr(getattr(temp_D11,item),'list_of_fields'):
                setattr(self,item,getattr(temp_D11,item))
        self.ref_surf.ref_pt_x_atc=x_atc_ctr
        self.ref_surf.rgt_azimuth=track_azimuth

    def select_ATL06_pairs(self, D6, pair_data):
        # based on data-quality flags in ATL06 data, and consistency checks
        # on the along-track and across-track slope in the data, select the 
        # highest-quality segments from the input data
        
        # ATBD section 5.1.2: select "valid pairs" for reference-surface calculation    
        # step 1a:  Select segs by data quality
        self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        self.pass_stats.ATL06_summary_zero_count=np.zeros((1,self.N_reps,))  # do we need to set to zeros?
        for cc in range(1,self.N_reps+1):
            if D6.cycle[D6.cycle==cc].shape[0] > 0:
                self.pass_stats.ATL06_summary_zero_count[0,cc-1]=np.sum(self.valid_segs.data[D6.cycle==cc])
                self.pass_stats.min_SNR_significance[0,cc-1]=np.amin(D6.snr_significance[D6.cycle==cc])
                self.pass_stats.min_signal_selection_source[0,cc-1]=np.amin(D6.signal_selection_source[D6.cycle==cc])
        self.ref_surf.N_pass_avail=np.count_nonzero(self.pass_stats.ATL06_summary_zero_count) 
        
        # step 1b: check if there are enough valid segments, quit if not
        if not np.any(self.valid_segs.data):
            self.status['atl06_quality_summary_all_nonzero']=1.0
            self.valid_segs.data[np.where(np.logical_or(D6.snr_significance<0.02, D6.signal_selection_source <=2))]=True
            if not np.any(self.valid_segs.data):
                self.status['atl06_quality_all_bad']=1
                return 
        
        # 1b: Select segs by height error        
        seg_sigma_threshold=np.maximum(self.params_11.seg_sigma_threshold_min, 3*np.median(D6.h_li_sigma[np.where(self.valid_segs.data)]))
        self.status['N_above_data_quality_threshold']=np.sum(D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data, D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data=np.logical_and( self.valid_segs.data , np.isfinite(D6.h_li_sigma))    
        
        # 1c: Map valid_segs.data to valid_pairs.data
        self.valid_pairs.data=np.logical_and(self.valid_segs.data[:,0], self.valid_segs.data[:,1])
        if not np.any(self.valid_pairs.data):
            self.status['no_valid_pairs']=1
            return 
        # 2a. see ATL06_pair.py
        # 2b. Calculate the y center of the slope regression
        self.y_polyfit_ctr=np.median(pair_data.y[self.valid_pairs.data])
        
        # 2c. identify segments close enough to the y center     
        self.valid_pairs.ysearch=np.abs(pair_data.y.ravel()-self.y_polyfit_ctr)<self.params_11.L_search_XT  
        
        # 3a: combine data and ysearch
        pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel()) 
        # 3b:choose the degree of the regression for across-track slope
        uX=np.unique(pair_data.x[pairs_valid_for_y_fit])
        if len(uX)>1 and uX.max()-uX.min() > 18.:
            my_regression_x_degree=1
        else:
            my_regression_x_degree=0
            
        uY=np.unique(pair_data.y[pairs_valid_for_y_fit])   
        if len(uY)>1 and uY.max()-uY.min() > 18.:
            my_regression_y_degree=1
        else:
            my_regression_y_degree=0
    
        # 3c: Calculate the formal error in the y slope estimates
        y_slope_sigma=np.sqrt(np.sum(D6.h_li_sigma[pairs_valid_for_y_fit,:]**2, axis=1))/np.transpose(np.diff(D6.y_atc[pairs_valid_for_y_fit,:], axis=1)).ravel() #same shape as y_slope*
        my_regression_tol=np.max(0.01, 3*np.median(y_slope_sigma))

        for iteration in range(2):
            # QUESTION: Do we need the "for item in range(2)" loop?  There are already 2 iterations in self.my_poly_fit.fit
            # 3d: regression of across-track slope against pair_data.x and pair_data.y
            self.my_poly_fit=poly_ref_surf(my_regression_x_degree, my_regression_y_degree, self.x_atc_ctr, self.y_polyfit_ctr) 
            y_slope_model, y_slope_resid,  y_slope_chi2r, y_slope_valid_flag=self.my_poly_fit.fit(pair_data.x[pairs_valid_for_y_fit], pair_data.y[pairs_valid_for_y_fit], D6.dh_fit_dy[pairs_valid_for_y_fit,0], max_iterations=1, min_sigma=my_regression_tol)
            # update what is valid based on regression flag
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=y_slope_valid_flag                #re-establish pairs_valid for y fit
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.y_slope.ravel()) # what about ysearch?
            
            # 3e: calculate across-track slope threshold
            if y_slope_resid.size>1:
                y_slope_threshold=np.max(my_regression_tol,3.*RDE(y_slope_resid))
            else:
                y_slope_threshold=my_regression_tol
            if ~pairs_valid_for_y_fit.any():
                pairs_valid_for_y_fit=np.zeros_like(pairs_valid_for_y_fit, dtype=bool)
                break
            # 3f: select for across-track residuals within threshold
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=np.abs(y_slope_resid)<=y_slope_threshold
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and( np.logical_and(self.valid_pairs.data.ravel(),self.valid_pairs.ysearch.ravel()), self.valid_pairs.y_slope.ravel()) 
                            
        # 3g. Use y model to evaluate all pairs
        self.valid_pairs.y_slope=np.abs(self.my_poly_fit.z(pair_data.x, pair_data.y)- pair_data.dh_dy) < y_slope_threshold 
        
        #4a. define pairs_valid_for_x_fit
        pairs_valid_for_x_fit= np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel())
        
        # 4b:choose the degree of the regression for along-track slope
        uX=np.unique(D6.x_atc[pairs_valid_for_x_fit,:].ravel())
        if len(uX)>1 and uX.max()-uX.min() > 18:
            mx_regression_x_degree=1
        else:
            mx_regression_x_degree=0
        uY=np.unique(D6.y_atc[pairs_valid_for_x_fit,:].ravel())    
        if len(uY)>1 and uY.max()-uY.min() > 18:
            mx_regression_y_degree=1
        else:
            mx_regression_y_degree=0

        #4c: Calculate along-track slope regression tolerance
        mx_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:].flatten())) 
        for iteration in range(2):
            # 4d: regression of along-track slope against x_pair and y_pair
            self.mx_poly_fit=poly_ref_surf(mx_regression_x_degree, mx_regression_y_degree, self.x_atc_ctr, self.y_polyfit_ctr) 
            if np.sum(pairs_valid_for_x_fit)>0:
                x_slope_model, x_slope_resid,  x_slope_chi2r, x_slope_valid_flag=self.mx_poly_fit.fit(D6.x_atc[pairs_valid_for_x_fit,:].ravel(), D6.y_atc[pairs_valid_for_x_fit,:].ravel(), D6.dh_fit_dx[pairs_valid_for_x_fit,:].ravel(), max_iterations=1, min_sigma=mx_regression_tol)
                # update what is valid based on regression flag
                x_slope_valid_flag.shape=[np.sum(pairs_valid_for_x_fit),2]
                self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=x_slope_valid_flag
                self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 

                # re-establish pairs_valid_for_x_fit
                pairs_valid_for_x_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.x_slope.ravel()) # include ysearch here?
            
                # 4e: calculate along-track slope threshold
                if x_slope_resid.size > 1.:
                    x_slope_threshold = np.max(mx_regression_tol,3*RDE(x_slope_resid))
                else:   
                    x_slope_threshold=mx_regression_tol

                # 4f: select for along-track residuals within threshold
                x_slope_resid.shape=[np.sum(pairs_valid_for_x_fit),2]
                self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=np.transpose(np.tile(np.all(np.abs(x_slope_resid)<=x_slope_threshold,axis=1),(2,1)))
                self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
                pairs_valid_for_x_fit=np.logical_and(np.logical_and(self.valid_pairs.data.ravel(),self.valid_pairs.ysearch.ravel()), self.valid_pairs.x_slope.ravel()) 

                if np.sum(pairs_valid_for_x_fit)==0:
                    self.status['no_valid_pairs']=1
            else:
                self.status['no_valid_pairs']=1
                
        # 4g. Use x model to evaluate all segments
        self.valid_segs.x_slope=np.abs(self.mx_poly_fit.z(D6.x_atc, D6.y_atc)- D6.dh_fit_dx) < x_slope_threshold #, max_iterations=2, min_sigma=mx_regression_tol)
        self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
    
        # 5: define selected pairs
        self.valid_pairs.all=np.logical_and(self.valid_pairs.data.ravel(), np.logical_and(self.valid_pairs.y_slope.ravel(), self.valid_pairs.x_slope.ravel()))
        if np.sum(self.valid_pairs.all)==0:
            self.status['no_valid_pairs']=1

        return
        
    def select_y_center(self, D6, pair_data):  #5.1.3
        cycle=D6.cycle[self.valid_pairs.all,:]
        # find the middle of the range of the selected beams
        y0=(np.min(D6.y_atc[self.valid_pairs.all,:].ravel())+np.max(D6.y_atc[self.valid_pairs.all,:].ravel()))/2
        # 1: define a range of y centers, select the center with the best score
        y0_shifts=np.round(y0)+np.arange(-100,100, 2)
        score=np.zeros_like(y0_shifts)

        # 2: search for optimal shift val.ue
        for count, y0_shift in enumerate(y0_shifts):
            sel_segs=np.all(np.abs(D6.y_atc[self.valid_pairs.all,:]-y0_shift)<self.params_11.L_search_XT, axis=1)
            sel_cycs=np.unique(cycle[sel_segs,0])
            selected_seg_cycle_count=len(sel_cycs)
            
            other_passes=np.unique(cycle.ravel()[~np.in1d(cycle.ravel(),sel_cycs)])
            unsel_segs=np.logical_and(np.in1d(cycle.ravel(),other_passes),np.abs(D6.y_atc[self.valid_pairs.all,:].ravel()-y0_shift)<self.params_11.L_search_XT)
            unsel_cycs=np.unique(cycle.ravel()[unsel_segs])
            unselected_seg_cycle_count=len(unsel_cycs)
            
            # the score is equal to the number of cycles with at least one valid pair entirely in the window, 
            # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
            score[count]=selected_seg_cycle_count + unselected_seg_cycle_count/100.
        # 3: identify the y0_shift value that corresponds to the best score, y_best, formally y_atc_ctr
        best = np.argwhere(score == np.amax(score))
        self.y_atc_ctr=np.median(y0_shifts[best])
        self.ref_surf.ref_pt_y_atc=self.y_atc_ctr
        
        if self.DOPLOT is not None and "score-vs-yshift" in self.DOPLOT:        
            plt.figure(2);plt.clf()
            plt.plot(y0_shifts,score,'.');
            plt.plot(np.ones_like(np.arange(1,np.amax(score)+1))*self.y_atc_ctr,np.arange(1,np.amax(score)+1),'r')
            plt.title('score vs y0_shifts(blu), y_best(red)')
        
        # 4: update valid pairs to inlucde y_atc within L_search_XT of y_atc_ctr (y_best)
        self.valid_pairs.ysearch=np.logical_and(self.valid_pairs.ysearch,np.abs(pair_data.y.ravel() - self.y_atc_ctr)<self.params_11.L_search_XT)  
        self.valid_pairs.all=np.logical_and(self.valid_pairs.ysearch.ravel(), self.valid_pairs.all.ravel())
        if self.DOPLOT is not None and "valid pair plot" in self.DOPLOT:
            plt.figure(50); plt.clf()
            plt.plot(pair_data.x, pair_data.y,'bo'); 
            plt.plot(pair_data.x[self.valid_pairs.data], pair_data.y[self.valid_pairs.data],'ro')
            plt.plot(D6.x_atc, D6.y_atc,'.');
            plt.plot(D6.x_atc[self.valid_pairs.all,:], D6.y_atc[self.valid_pairs.all,:],'+');
            plt.grid(True)
        if self.DOPLOT is not None and "valid pair plot" in self.DOPLOT:     
            plt.figure(51);plt.clf()
            plt.plot(np.abs(pair_data.y - self.y_atc_ctr)<self.params_11.L_search_XT[self.valid_pairs.data])

        return 

    def find_reference_surface(self, D6, DEBUG=None):  #5.1.4

        self.corrected_h.quality_summary=np.logical_not(np.logical_and( np.logical_and(self.pass_stats.min_signal_selection_source<=1, self.pass_stats.min_SNR_significance<0.02),self.pass_stats.ATL06_summary_zero_count>0 ))        
        self.ref_surf.complex_surface_flag=0

        # establish attributes: all possible exponent combinations, in order, for reference surface fit 
        self.non_product.poly_exponent_x=np.array([])
        for jj in range(params_11.poly_max_degree_AT):
            self.non_product.poly_exponent_x=np.append(self.non_product.poly_exponent_x,np.arange(jj+1,-1,-1),axis=0).astype('int')        
        self.non_product.poly_exponent_y=np.array([])
        for jj in range(params_11.poly_max_degree_XT):
            self.non_product.poly_exponent_y=np.append(self.non_product.poly_exponent_y,np.arange(jj+2),axis=0).astype('int')
        self.ref_surf.surf_fit_quality_summary=0
        
        # in this section we only consider segments in valid pairs
        self.selected_segments=np.column_stack( (self.valid_pairs.all,self.valid_pairs.all) )
        # Table 4-2        
        self.pass_stats.pass_seg_count=np.zeros((1,self.N_reps,))
        self.pass_stats.pass_included_in_fit=np.zeros((1,self.N_reps,))

        # establish new boolean arrays for selecting
        selected_pairs=np.ones( (np.sum(self.valid_pairs.all),),dtype=bool) 
        selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()  

        cycle=D6.cycle[self.valid_pairs.all,:].ravel()  # want the cycle of each seg in valid pair
        self.ref_surf_passes = np.unique(cycle) 

        # 1. build cycle design matrix with selected segments (those in valid_pairs, initially)
        data=np.ones(len(cycle))
        row=np.array([],dtype=int)
        col=np.array([],dtype=int)
        for index, item in enumerate(self.ref_surf_passes):
            row=np.append(row,np.nonzero(cycle==item))
            col=np.append(col,np.array(index*np.ones(np.count_nonzero(cycle==item))))
        G_Zp=sparse.csc_matrix((data,(row,col)),shape=[len(cycle),len(self.ref_surf_passes)])    
        self.repeat_cols=np.arange(G_Zp.toarray().shape[1])
        
        # 2. determine polynomial degree, using unique x's and unique y's of segments in valid pairs
        x_atcU = np.unique(D6.x_atc[self.valid_pairs.all,:].ravel()) # np.unique orders the unique values
        y_atcU = np.unique(D6.y_atc[self.valid_pairs.all,:].ravel()) # np.unique orders the unique values
        # Table 4-4
        self.ref_surf.n_deg_x = np.minimum(self.params_11.poly_max_degree_AT,len(x_atcU)-1) 
        self.ref_surf.n_deg_y = np.minimum(self.params_11.poly_max_degree_XT,len(y_atcU)-1) 
        # 3. perform an iterative fit for the across track polynomial
        # 3a. define degree_list_x and degree_list_y.  These are stored in self.default.poly_exponent_list      
        degree_x=np.array([item[0] for item in self.params_11.poly_exponent_list], dtype=int)
        degree_y=np.array([item[1] for item in self.params_11.poly_exponent_list], dtype=int)
        # keep only degrees > 0 and degree_x+degree_y <= max(max_x_degree, max_y_degree)
        self.poly_mask=(degree_x + degree_y) <= np.maximum(self.ref_surf.n_deg_x,self.ref_surf.n_deg_y)
        #print(self.degree_list_x,self.degree_list_x.shape)
        self.degree_list_x = degree_x[self.poly_mask]
        self.degree_list_y = degree_y[self.poly_mask]

        # 3b. define polynomial matrix
        if DEBUG:        
            print('x_ctr is',self.x_atc_ctr)
            print('y_ctr is',self.y_atc_ctr)
        x_atc=D6.x_atc[self.valid_pairs.all,:].ravel()
        y_atc=D6.y_atc[self.valid_pairs.all,:].ravel()
        S_fit_poly=np.zeros((len(x_atc),len(self.degree_list_x)),dtype=float)
        for jj in range(len(x_atc)):
            for ii in range(len(self.degree_list_x)):
                x_term=( (x_atc[jj]-self.x_atc_ctr)/self.params_11.xy_scale )**self.degree_list_x[ii]
                y_term=( (y_atc[jj]-self.y_atc_ctr)/self.params_11.xy_scale )**self.degree_list_y[ii]
                S_fit_poly[jj,ii]=x_term*y_term                
            
        # 3c. define slope-change matrix 
        # 3d. build the fitting matrix
        delta_time=D6.delta_time[self.valid_pairs.all,:].ravel()

        if self.slope_change_t0/self.params_11.t_scale > 1.5:
            x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            self.G_surf=np.concatenate( (S_fit_poly,S_fit_slope_change,G_Zp.toarray()),axis=1 ) # G = [S St D]
            self.poly_cols=np.arange(S_fit_poly.shape[1])
            self.slope_change_cols=np.arange(S_fit_slope_change.shape[1])
        else:
            self.G_surf=np.concatenate( (S_fit_poly,G_Zp.toarray()),axis=1 ) # G = [S D]
            self.poly_cols=np.arange(S_fit_poly.shape[1])
            self.slope_change_cols=np.array([])
            
        fit_columns=np.ones(self.G_surf.shape[1],dtype=bool)

        # reduce these variables to the segments of the valid pairs only
        h_li_sigma=D6.h_li_sigma[self.valid_pairs.all,:].ravel() 
        h_li      =D6.h_li[self.valid_pairs.all,:].ravel()
            
        for kk in range(self.params_11.max_fit_iterations): 
             
            G=self.G_surf[selected_segs,:]
            # 3e. If more than one repeat is present, subset 
            #fitting matrix, to include columns that are not uniform
            if self.ref_surf_passes.size > 1:
                for c in range(G.shape[1]-1,-1,-1):   # check lat col first, do in reverse order
                    if np.all(G[:,c]==G[0,c]):
                        fit_columns[c]=False
                G=G[:, fit_columns]
            else:
                self.ref_surf.surf_fit_quality_summary=1
                
            # if three or more cycle columns are lost, use planar fit in x and y (end of section 3.3)
            if np.sum(np.logical_not(fit_columns[np.sum([self.poly_cols.shape,self.slope_change_cols.shape]):,])) > 2: 
                print('you are in olanar')
                S_fit_poly=np.zeros((len(x_atc),2),dtype=float)
                for jj in range(len(x_atc)):
                    x_term=( (x_atc[jj]-self.x_atc_ctr)/self.params_11.xy_scale )
                    y_term=( (y_atc[jj]-self.y_atc_ctr)/self.params_11.xy_scale )
                    S_fit_poly[jj,0]=x_term
                    S_fit_poly[jj,1]=y_term
                self.poly_cols=np.arange(S_fit_poly.shape[1])
                self.slope_change_cols=np.array([])
                self.G_surf=np.concatenate( (S_fit_poly,G_Zp.toarray()),axis=1 )
        
                # re-make mask for polynomial coefficients
                self.degree_list_x=np.array([1])
                self.degree_list_y=np.array([1])
                degree_list=np.transpose(np.vstack((self.degree_list_x,self.degree_list_y))).tolist()
                self.poly_mask=[False]*len(poly_exponent)
                for ii, item in enumerate(degree_list):
                    self.poly_mask[poly_exponent.index(degree_list[ii])]=True

                fit_columns=np.ones(self.G_surf.shape[1],dtype=bool)
                Cd, Cdi, G_g = gen_inv(self,self.G_surf,h_li_sigma)
                self.m_surf=np.zeros(np.size(self.G_surf,1))            
                self.m_surf=np.dot(G_g,h_li)  
                selected_pairs=np.ones( (np.sum(self.valid_pairs.all),),dtype=bool) 
                selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
                r_seg=h_li-np.dot(self.G_surf,self.m_surf)
                r_fit=r_seg
                self.ref_surf.complex_surface_flag=1
                break
            # 3f. generate the data-covariance matrix
            # 3g. equation 7
            Cd, Cdi, G_g = gen_inv(self,G,h_li_sigma[selected_segs])
                        
            # inititalize the reference model 
            self.m_surf=np.zeros(np.size(self.G_surf,1))            
            z=h_li[selected_segs]            
            self.m_surf[fit_columns]=np.dot(G_g,z)  

            # 3h. Calculate model residuals for all segments
            r_seg=h_li-np.dot(self.G_surf,self.m_surf)            
            r_fit=r_seg[selected_segs] 
            
            # 3i. Calculate the fitting tolerance, 
            r_tol = 3*RDE(r_fit/h_li_sigma[selected_segs])
            # reduce chi-squared value
            self.ref_surf.surf_fit_misfit_chi2 = np.dot(np.dot(np.transpose(r_fit),Cdi.toarray()),r_fit)

            # calculate P value
            n_cols=np.sum(fit_columns)
            n_rows=np.sum(selected_segs)
            P = 1 - stats.chi2.cdf(self.ref_surf.surf_fit_misfit_chi2, n_rows-n_cols)

            # 3j. 
            if P<0.025 and kk < self.params_11.max_fit_iterations-1:
                selected_segs_prev=selected_segs
                selected_segs = np.abs(r_seg/h_li_sigma) < r_tol # boolean
                if np.all( selected_segs_prev==selected_segs ):
                    break
            
            # make selected_segs pair-wise consistent        
            selected_pairs=selected_segs.reshape((len(selected_pairs),2)).all(axis=1)  
            selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
            if P>0.025:
                break
  
        self.ref_surf.surf_fit_misfit_RMS=RDE(r_fit)   # Robos Dispersion Estimate, half the diff bet the 16th and 84th percentiles of a distribution
        self.selected_segments[np.nonzero(self.selected_segments)]=selected_segs

        if self.slope_change_cols.shape[0]>0:
            self.ref_surf_passes=self.ref_surf_passes[fit_columns[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols]]            
        else:
            self.ref_surf_passes=self.ref_surf_passes[fit_columns[self.poly_cols.shape[0]+self.repeat_cols]]
    
        # report the selected segments 
        selected_pair_out= self.valid_pairs.all.copy()
        selected_pair_out[selected_pair_out==True]=selected_segs.reshape((len(selected_pairs),2)).all(axis=1)
        
        self.valid_pairs.iterative_fit=selected_pair_out
        
        self.valid_segs.iterative_fit=np.column_stack((self.valid_pairs.iterative_fit, self.valid_pairs.iterative_fit))
        
        if self.DOPLOT is not None and "3D time plot" in self.DOPLOT:
            x_atc = D6.x_atc[self.selected_segments]       
            y_atc = D6.y_atc[self.selected_segments]
            h_li  = D6.h_li[self.selected_segments]        
            h_li_sigma = D6.h_li_sigma[self.selected_segments]
            fig=plt.figure(31); plt.clf(); ax=fig.add_subplot(111, projection='3d')        
            p=ax.scatter(x_atc, y_atc, h_li, c=time); 
            fig.colorbar(p)
            fig=plt.figure(32); plt.clf(); ax=fig.add_subplot(111, projection='3d')
            p=ax.scatter(x_atc, y_atc, h_li, c=np.abs(r_seg[selected_segs]/h_li_sigma)) 
            fig.colorbar(p)
        
        # separate self.m_surf
        self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)]=self.m_surf[self.poly_cols]

        if self.slope_change_cols.shape[0]>0:
            self.ref_surf.slope_change_rate_x=np.array([self.m_surf[self.poly_cols.shape[0]+self.slope_change_cols[0]]])
            self.ref_surf.slope_change_rate_y=np.array([self.m_surf[self.poly_cols.shape[0]+self.slope_change_cols[1]]])
            self.m_ref=np.concatenate( ([self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)].ravel(),self.ref_surf.slope_change_rate_x,self.ref_surf.slope_change_rate_y]), axis=0)
        else:
            self.ref_surf.slope_change_rate_x=np.nan
            self.ref_surf.slope_change_rate_y=np.nan
            self.m_ref=self.m_surf[self.poly_cols]
        # check if slope change rate for either x or y is > 0.1 (see Table 4-4)     
        if np.any((np.logical_or(np.abs(self.ref_surf.slope_change_rate_x)>0.1,np.isnan(self.ref_surf.slope_change_rate_x)),
                   np.logical_or(np.abs(self.ref_surf.slope_change_rate_y)>0.1,np.isnan(self.ref_surf.slope_change_rate_y)))):
            self.ref_surf.surf_fit_quality_summary=1

        self.z_cycle=self.m_surf[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols] # the 'intercept'
        
        if self.slope_change_cols.shape[0]>0:
            self.corrected_h.pass_h_shapecorr[0,self.ref_surf_passes.astype(int)-1]=self.z_cycle[fit_columns[np.sum([self.poly_cols.shape,self.slope_change_cols.shape]):]] 
        else:
            self.corrected_h.pass_h_shapecorr[0,self.ref_surf_passes.astype(int)-1]=self.z_cycle[fit_columns[self.poly_cols.shape[0]:]] 

        # get terms of pass_h_shapecorr_sigma_systematic, equation 12
        term1=(D6.dh_fit_dx * D6.sigma_geo_at)**2
        # use the reference surface slope for the y component
        term2=(D6.dh_fit_dy * D6.sigma_geo_xt)**2
        term3=(D6.sigma_geo_h)**2

        h_li_sigma = D6.h_li_sigma[self.selected_segments]
        cycle      = D6.cycle[self.selected_segments]
        for cc in self.ref_surf_passes.astype(int):
            cycle_segs=np.flatnonzero(self.selected_segments)[cycle==cc]
            self.corrected_h.mean_pass_time[0,cc-1]       =np.mean(D6.delta_time.ravel()[cycle_segs])            
            self.pass_stats.pass_included_in_fit[0,cc-1]=1
            self.pass_stats.pass_seg_count[0, cc-1]=cycle_segs.size            
            self.pass_stats.mean_pass_lon[0,cc-1]         =np.mean(D6.longitude.ravel()[cycle_segs])
            self.pass_stats.mean_pass_lat[0,cc-1]         =np.mean(D6.latitude.ravel()[cycle_segs])
            self.pass_stats.x_atc_mean[0,cc-1]            =np.mean(D6.x_atc.ravel()[cycle_segs])
            self.pass_stats.y_atc_mean[0,cc-1]            =np.mean(D6.y_atc.ravel()[cycle_segs])
            self.pass_stats.cloud_flg_asr_best[0,cc-1]    =np.min(D6.cloud_flg_asr.ravel()[cycle_segs])
            self.pass_stats.cloud_flg_atm_best[0,cc-1]    =np.min(D6.cloud_flg_atm.ravel()[cycle_segs])
            self.pass_stats.bsnow_conf_best[0,cc-1]       =np.max(D6.bsnow_conf.ravel()[cycle_segs])
            # weighted means
            W_by_error=h_li_sigma[cycle==cc]**(-2)/np.sum(h_li_sigma[cycle==cc]**(-2))
            self.pass_stats.bsnow_h_mean[0,cc-1]        =np.sum( W_by_error * D6.bsnow_h.ravel()[cycle_segs] )         
            self.pass_stats.r_eff_mean[0,cc-1]          =np.sum( W_by_error * D6.r_eff.ravel()[cycle_segs] )           
            self.pass_stats.tide_ocean_mean[0,cc-1]     =np.sum( W_by_error * D6.tide_ocean.ravel()[cycle_segs] )       
            self.pass_stats.h_robust_spread_mean[0,cc-1]=np.sum( W_by_error * D6.h_robust_spread.ravel()[cycle_segs] )  
            self.pass_stats.h_li_rms_mean[0,cc-1]       =np.sum( W_by_error * D6.h_rms_misft.ravel()[cycle_segs] )     
            self.pass_stats.sigma_geo_h_mean[0,cc-1]    =np.sqrt( np.sum( W_by_error * D6.sigma_geo_h.ravel()[cycle_segs]**2 ) )     
            self.pass_stats.sigma_geo_at_mean[0,cc-1]   =np.sqrt( np.sum( W_by_error * D6.sigma_geo_at.ravel()[cycle_segs]**2 ) )    
            self.pass_stats.sigma_geo_xt_mean[0,cc-1]   =np.sqrt( np.sum( W_by_error * D6.sigma_geo_xt.ravel()[cycle_segs]**2 ) )    
            self.corrected_h.pass_h_shapecorr_sigma_systematic[0,cc-1] =\
                np.sqrt(np.sum(W_by_error*(term1.ravel()[cycle_segs] + term2.ravel()[cycle_segs] + term3.ravel()[cycle_segs])))
            self.pass_stats.h_uncorr_mean[0,cc-1]=np.sum( W_by_error * D6.h_li.ravel()[cycle_segs] )

        self.ref_surf.N_pass_used=np.count_nonzero(self.ref_surf_passes)

        if self.ref_surf.N_pass_used<2:
            self.ref_surf.surf_fit_quality_summary=1
        
        # 3k. propagate the errors   
        Cdp=sparse.diags(np.maximum(h_li_sigma**2,(RDE(r_fit))**2))  # C1 in text  
        self.C_m_surf = np.dot(np.dot(G_g,Cdp.toarray()),np.transpose(G_g))
        self.m_surf_sigma=np.full(self.G_surf.shape[1],np.nan)
        self.m_surf_sigma[fit_columns]=np.sqrt(self.C_m_surf.diagonal())
        self.ref_surf.poly_coeffs_sigma[0,np.where(self.poly_mask)]=self.m_surf_sigma[self.poly_cols]
        if self.slope_change_cols.shape[0]>0:          
            self.ref_surf.slope_change_rate_x_sigma=self.m_surf_sigma[self.poly_cols.shape[0]+self.slope_change_cols[0]]
            self.ref_surf.slope_change_rate_y_sigma=self.m_surf_sigma[self.poly_cols.shape[0]+self.slope_change_cols[1]]
        else:
            self.ref_surf.slope_change_rate_x_sigma=np.full((1,),np.nan)
            self.ref_surf.slope_change_rate_y_sigma=np.full((1,),np.nan)
        
        if self.DOPLOT is not None and "plot of polynomial degree and errors" in self.DOPLOT:
            plt.figure(3);plt.clf()
            plt.plot(self.m_surf_sigma[:np.sum(self.poly_cols.shape,self.slope_change_cols.shape)],'ro-')
            plt.hold(True)
            plt.plot(self.m_ref[:np.sum(self.poly_cols.shape,self.slope_change_cols.shape)],'go-')
            plt.xticks(np.arange(9),(self.degree_list_x+self.degree_list_y).astype('S3'))
            plt.xlabel('sum of x_degree, y degree')
            plt.title('Surface Shape Polynomial (g), Sigma m (r)')
        self.z_cycle_sigma=self.m_surf_sigma[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols] # the 'intercept'

        self.corrected_h.pass_h_shapecorr_sigma[0,self.ref_surf_passes.astype(int)-1]=self.z_cycle_sigma

        # convert slope change rates to 1/years
        self.ref_surf.slope_change_rate_x = self.ref_surf.slope_change_rate_x  * self.params_11.t_scale  
        self.ref_surf.slope_change_rate_y = self.ref_surf.slope_change_rate_y  * self.params_11.t_scale  
        
        # calculate fit slopes and curvature
        northing=np.arange(-50., 60,10);
        easting =np.arange(-50., 60,10);
        [N,E]=np.meshgrid(northing,easting)
        cos_az=np.cos(self.ref_surf.rgt_azimuth*np.pi/180) 
        sin_az=np.sin(self.ref_surf.rgt_azimuth*np.pi/180) 
        
        xg=N*cos_az + E*sin_az
        if self.DOPLOT is not None and "NE-vs-xy" in self.DOPLOT:
            plt.figure(101);plt.clf()
            plt.imshow( (xg-self.x_atc_ctr)/self.params_11.xy_scale);plt.colorbar()
        
        yg=-N*sin_az + E*cos_az
        if self.DOPLOT is not None and "NE-vs-xy" in self.DOPLOT:
            plt.figure(102);plt.clf()
            plt.imshow( (yg-self.y_atc_ctr)/self.params_11.xy_scale);plt.colorbar()
        
        zg=np.zeros_like(xg)
        for ii in np.arange(np.sum(self.poly_mask)):
            xterm=( xg/self.params_11.xy_scale )**self.degree_list_x[ii]
            yterm=( yg/self.params_11.xy_scale )**self.degree_list_y[ii]
            zg=zg+self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)][0,ii] * xterm * yterm 
        if self.DOPLOT is not None and "NE-vs-xy" in self.DOPLOT:
            plt.figure(100);plt.clf()
            plt.contourf(zg);plt.colorbar()

        # fitting a plane as a function of N and E 
        M=np.transpose(np.vstack(( (N.ravel()),(E.ravel()), np.ones_like(E.ravel()))))
        msub,rr,rank,sing=linalg.lstsq(M,zg.ravel())
        zg_plane=msub[0]*N + msub[1]*E;
        
        # perform the same fit in [xg,yg] to calculate the y slope for the unselected segments
        M_xy=np.transpose(np.vstack(( (xg.ravel()),(yg.ravel()), np.ones_like(xg.ravel()))))
        msub_xy,rr, rankxy, singxy=linalg.lstsq(M_xy, zg.ravel())
        self.ref_surf_slope_x=msub_xy[0]
        self.ref_surf.fit_y_slope=msub_xy[1]
        
        if self.DOPLOT is not None and "NE-vs-xy" in self.DOPLOT:
            plt.figure(104);plt.clf()
            plt.contourf(zg_plane);plt.colorbar()
        self.ref_surf.fit_N_slope=msub[0] 
        self.ref_surf.fit_E_slope=msub[1] 
        self.ref_surf.fit_curvature=rr
        
        return 
    
    def corr_heights_other_cycles(self, D6):
        # find cycles not in ref_surface_passes 
        other_passes=np.unique(D6.cycle.ravel()[~np.in1d(D6.cycle.ravel(),self.ref_surf_passes)])
        # 1. find cycles not in ref_surface_passes, but have valid_segs.data and valid_segs.x_slope  
        non_ref_segments=np.logical_and(np.in1d(D6.cycle.ravel(),other_passes),np.logical_and(self.valid_segs.data.ravel(),self.valid_segs.x_slope.ravel()))
        #  If the x polynomial degree is zero, allow only segments that have x_atc matching that of the valid segments (+- 10 m)
        if (self.degree_list_x==0).all():
            ref_surf_x_ctrs=D6.x_atc[self.selected_segments]
            ref_surf_x_range=np.array([ref_surf_x_ctrs.min(), ref_surf_x_ctrs.max()])
            non_ref_segments=np.logical_and(non_ref_segments, D6.x_atc.ravel() > ref_surf_x_range[0]-10.)
            non_ref_segments=np.logical_and(non_ref_segments, D6.x_atc.ravel() < ref_surf_x_range[1]+10.)
        
        if non_ref_segments.any():
            # 2. build design matrix, G_other, for non selected segments (poly and dt parts only)
            x_atc=D6.x_atc.ravel()[non_ref_segments]
            y_atc=D6.y_atc.ravel()[non_ref_segments]
            if self.ref_surf.complex_surface_flag==0:
                S_fit_poly=np.zeros((len(x_atc),len(self.degree_list_x)),dtype=float)
                for jj in range(len(x_atc)):
                    for ii in range(len(self.degree_list_x)):
                        x_term=( (x_atc[jj]-self.x_atc_ctr)/self.params_11.xy_scale )**self.degree_list_x[ii]
                        y_term=( (y_atc[jj]-self.y_atc_ctr)/self.params_11.xy_scale )**self.degree_list_y[ii]
                        S_fit_poly[jj,ii]=x_term*y_term            
            else:
                S_fit_poly=np.zeros((len(x_atc),2),dtype=float)
                for jj in range(len(x_atc)):
                    x_term=( (x_atc[jj]-self.x_atc_ctr)/self.params_11.xy_scale )
                    y_term=( (y_atc[jj]-self.y_atc_ctr)/self.params_11.xy_scale )
                    S_fit_poly[jj,0]=x_term
                    S_fit_poly[jj,1]=y_term
            
            delta_time=D6.delta_time.ravel()[non_ref_segments]
            if self.slope_change_cols.shape[0]>0:
                x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
                y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
                S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
                G_other=np.concatenate( (S_fit_poly,S_fit_slope_change),axis=1 ) # G [S St]
                surf_model=np.concatenate((self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)].ravel(),self.ref_surf.slope_change_rate_x,self.ref_surf.slope_change_rate_y),axis=0)
            else:
                G_other=S_fit_poly  #  G=[S]
                surf_model=np.transpose(self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)])
         
            # heights and errors from non_ref_segments 
            h_li      =D6.h_li.ravel()[non_ref_segments]
            h_li_sigma=D6.h_li_sigma.ravel()[non_ref_segments]
            cycle=D6.cycle.ravel()[non_ref_segments]
            
            self.non_ref_surf_passes=np.unique(cycle)
            # section 3.5
            # calculate corrected heights, z_kc, with non selected segs design matrix and surface shape polynomial from selected segments
            z_kc=h_li - np.dot(G_other,surf_model).ravel()
            if self.DOPLOT is not None and "h_corr other cycles_plot" in self.DOPLOT:
                plt.figure(107);plt.clf()
                plt.plot(h_li,'b.-');plt.hold(True)
                plt.plot(z_kc,'ro-')
                plt.title('Other cycles: hli (b), Zkc-Corrected Heights (r)');plt.grid()
            
            # use errors from surface shape polynomial and non-selected segs design matrix to get non selected segs height corrs errors
            if self.slope_change_cols.shape[0]>0:
                Cms=self.C_m_surf[:,np.concatenate( (self.poly_cols,self.slope_change_cols) )][np.concatenate( (self.poly_cols,self.slope_change_cols) ),:] 
            else:
                Cms=self.C_m_surf[:,self.poly_cols][self.poly_cols,:] # can't index 2 dimensions at once. 
            z_kc_sigma = np.sqrt( np.diag( np.dot(np.dot(G_other,Cms),np.transpose(G_other)) ) + h_li_sigma**2 ) # equation 11

            if self.DOPLOT is not None and "h_corr other cycles_plot" in self.DOPLOT:       
                plt.figure(108);plt.clf()
                plt.plot(h_li_sigma,'b.-');plt.hold(True)
                plt.plot(z_kc_sigma,'ro-')
                plt.title('Other cycles: hli sigma (b), Zkc sigma(r)');plt.grid()
            
            # get terms of pass_h_shapecorr_sigma_systematic, equation 12
            term1=(D6.dh_fit_dx * D6.sigma_geo_at)**2
            term2=(self.ref_surf.fit_y_slope * D6.sigma_geo_xt)**2
            term3=(D6.sigma_geo_h)**2
            
            for cc in self.non_ref_surf_passes.astype(int):
                best_seg=np.argmin(z_kc_sigma[cycle==cc])
                best_seg_ind=np.flatnonzero(non_ref_segments)[cycle==cc][best_seg]
                self.corrected_h.pass_h_shapecorr[0,cc-1]      =z_kc[cycle==cc][best_seg]
                self.corrected_h.pass_h_shapecorr_sigma[0,cc-1]= z_kc_sigma[cycle==cc][best_seg]
                self.corrected_h.pass_h_shapecorr_sigma_systematic[0,cc-1] = \
                    np.sqrt(term1.ravel()[best_seg_ind] + term2.ravel()[best_seg_ind]  + term3.ravel()[best_seg_ind])
                self.corrected_h.mean_pass_time[0,cc-1]        =D6.delta_time.ravel()[best_seg_ind]
                self.pass_stats.pass_seg_count[0,cc-1]         =1
                self.pass_stats.mean_pass_lon[0,cc-1]          =D6.longitude.ravel()[best_seg_ind]
                self.pass_stats.mean_pass_lat[0,cc-1]          =D6.latitude.ravel()[best_seg_ind]
                self.pass_stats.x_atc_mean[0,cc-1]             =D6.x_atc.ravel()[best_seg_ind]
                self.pass_stats.y_atc_mean[0,cc-1]             =D6.y_atc.ravel()[best_seg_ind]
                self.pass_stats.bsnow_h_mean[0,cc-1]           =D6.bsnow_h.ravel()[best_seg_ind] 
                self.pass_stats.r_eff_mean[0,cc-1]             =D6.r_eff.ravel()[best_seg_ind]
                self.pass_stats.tide_ocean_mean[0,cc-1]        =D6.tide_ocean.ravel()[best_seg_ind]
                self.pass_stats.h_robust_spread_mean[0,cc-1]   =D6.h_robust_spread.ravel()[best_seg_ind]
                self.pass_stats.h_li_rms_mean[0,cc-1]          =D6.h_rms_misft.ravel()[best_seg_ind]
                self.pass_stats.sigma_geo_h_mean[0,cc-1]       =D6.sigma_geo_h.ravel()[best_seg_ind]
                self.pass_stats.sigma_geo_at_mean[0,cc-1]      =D6.sigma_geo_at.ravel()[best_seg_ind]
                self.pass_stats.sigma_geo_xt_mean[0,cc-1]      =D6.sigma_geo_xt.ravel()[best_seg_ind]
                self.pass_stats.h_uncorr_mean[0, cc-1]         =D6.h_li.ravel()[best_seg_ind]
            # establish segment_id_by_cycle for selected segments from reference surface finding and for non_ref_surf
            self.segment_id_by_cycle=[]         
            self.selected_segments_by_cycle=[]         
            cyc=D6.cycle[self.selected_segments[:,0],0]  
            segid=D6.segment_id[self.selected_segments[:,0],0]    
        
            non_cyc=D6.cycle.ravel()[non_ref_segments]  
            non_segid=D6.segment_id.ravel()[non_ref_segments]  
        
            for cc in range(1,13):  
                if np.in1d(cc,self.ref_surf_passes):
                    self.segment_id_by_cycle.append( np.array( segid[cyc==cc] ) )
                elif np.in1d(cc,self.non_ref_surf_passes):
                    self.segment_id_by_cycle.append( np.array( non_segid[non_cyc==cc] ) )
                else:     
                    self.segment_id_by_cycle.append(np.array([]))

            self.selected_segments=np.logical_or(self.selected_segments,non_ref_segments.reshape(self.valid_pairs.all.shape[0],2))

            if self.DOPLOT is not None and "select best error by cycle" in self.DOPLOT:
                plt.figure(200);plt.clf()
                plt.plot(np.arange(12)+1,self.corrected_h.pass_h_shapecorr,'bo-');plt.hold(True)
                plt.plot(np.arange(12)[self.non_ref_surf_passes.astype(int)-1]+1,self.corrected_h.pass_h_shapecorr[self.non_ref_surf_passes.astype(int)-1],'ro')
                plt.xlabel('Cycle Number');plt.xlim((0,13))
                plt.ylabel('Corrected Height with lowest Error / Cycle')
                plt.title('Pass H ShapeCorr: selected (b), other (r)');plt.grid()
                plt.figure(201);plt.clf()
                plt.plot(np.arange(12)+1,self.corrected_h.pass_h_shapecorr_sigma,'bo-');plt.hold(True)
                plt.plot(np.arange(12)[self.non_ref_surf_passes.astype(int)-1]+1,self.corrected_h.pass_h_shapecorr_sigma[self.non_ref_surf_passes.astype(int)-1],'ro')
                plt.xlabel('Cycle Number');plt.xlim((0,13))
                plt.ylabel('Lowest Error of Segments in each Cycle')
                plt.title('Pass H ShapeCorr Sigma: selected (b), other (r)');plt.grid()
        
def gen_inv(self,G,sigma):
    # 3f. Generate data-covariance matrix
    Cd=sparse.diags(sigma**2)
    Cdi=sparse.diags(1/sigma**2)
    G_sq=np.dot(np.dot(np.transpose(G),Cdi.toarray()),G)
            
    G_sqi=linalg.inv(G_sq)
    # calculate the generalized inverse of G
    G_g=np.dot( np.dot(G_sqi,np.transpose(G)),Cdi.toarray() )  
            
    return Cd, Cdi, G_g
        
         
class ATL11_defaults:
    def __init__(self):
        # provide option to read keyword=val pairs from the input file
        self.L_search_AT=100 # meters, along track (in x), filters along track
        self.L_search_XT=110 # meters, cross track (in y), filters across track
        #self.min_slope_tol=0.02 # in degrees?
        #self.min_h_tol=0.1  # units? of height offset
        self.seg_sigma_threshold_min=0.05
        self.beam_spacing=90 # meters
        self.seg_atc_spacing=100 # meters
        self.N_search=self.L_search_AT/20 # segments to include +/- from seg_atc_ctr in each ref_pt analysis
        self.poly_max_degree_AT=3
        self.poly_max_degree_XT=3
        self.xy_scale=100.         # meters
        self.t_scale=86400*365.25  # seconds/year.
        self.max_fit_iterations = 20  # maximum iterations when computing the reference surface models
        self.equatorial_radius=6378137 # meters, on WGS84 spheroid
        self.polar_radius=6356752.3 # derived, https://www.eoas.ubc.ca/~mjelline/Planetary%20class/14gravity1_2.pdf
        
        # calculate the order for the polynomial degrees:  Sorted by degree, then by y degree, no sum of x and y degrees larger than max(degree_x, degree_y)
        degree_list_x, degree_list_y = np.meshgrid(np.arange(self.poly_max_degree_AT+1), np.arange(self.poly_max_degree_XT+1))
        # keep only degrees > 0 and degree_x+degree_y <= max(max_x_degree, max_y_degree)
        sum_degrees=( degree_list_x +  degree_list_y).ravel()
        keep=np.where(np.logical_and( sum_degrees <= np.maximum(self.poly_max_degree_AT,self.poly_max_degree_XT), sum_degrees > 0 ))
        degree_list_x = degree_list_x.ravel()[keep]
        degree_list_y = degree_list_y.ravel()[keep]
        sum_degree_list = sum_degrees[keep]
        # order by sum, x and then y
        degree_order=np.argsort(sum_degree_list + (degree_list_y / (degree_list_y.max()+1)))
        self.poly_exponent_list=np.transpose(np.vstack((degree_list_x[degree_order], degree_list_y[degree_order]))).tolist()
        self.N_coeffs=len(self.poly_exponent_list)
     
     
     
