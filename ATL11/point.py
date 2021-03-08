# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import pointCollection as pc
#from poly_ref_surf import poly_ref_surf
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import linalg
from scipy import stats
import ATL11

class point(ATL11.data):
    # ATL11_point is a class with methods for calculating ATL11 from ATL06 data
    def __init__(self, N_pairs=1, ref_pt=None, beam_pair=None, x_atc_ctr=np.NaN,  track_azimuth=np.NaN, max_poly_degree=[1, 1], cycles=[1,12],  rgt=None, mission_time_bds=None, params_11=None):
        # input variables:
        # N_pairs: Number of distinct pairs in the ATL06 data
        # ref_pt: the reference-point number for the ATL11 fit.  This is the geoseg number for the central segment of the fit
        # x_atc_ctr: the along-track corresponding to ref_pt
        # track_azimuth: the azimuth of the RGT for the ATL11 point
        # optional parameters:
        # max_poly_degree: the maximum degree of the along- and across-track polynomials
        # cycles: first and last repeats that might appear in the ATL06 data
        # mission_time_bnds: The start and end of the mission, in delta_time units (seconds)
        # params_11: ATL11_defaults structure

        if params_11 is None:
            self.params_11=ATL11.defaults()
        else:
            self.params_11=params_11
        # initialize the data structure using the ATL11_data __init__ method
        ATL11.data.__init__(self,N_pts=1, cycles=cycles, N_coeffs=self.params_11.N_coeffs)
        self.N_pairs=N_pairs
        self.cycles=cycles
        self.N_coeffs=self.params_11.N_coeffs
        self.x_atc_ctr=x_atc_ctr
        self.beam_pair=beam_pair
        self.ref_pt=ref_pt
        self.track_azimuth=track_azimuth
        self.ref_surf_slope_x=np.NaN
        self.ref_surf_slope_y=np.NaN
        self.calc_slope_change=False
        self.rgt=rgt
        if mission_time_bds is None:
            mission_time_bds=np.array([0, cycles[1]*91*24*3600])
        self.slope_change_t0=mission_time_bds[0]+0.5*(mission_time_bds[1]-mission_time_bds[0])
        self.mission_time_bds=mission_time_bds
        self.valid_segs =ATL11.validMask((N_pairs,2), ('data','x_slope' ))  #  2 cols, boolan, all F to start
        self.valid_pairs=ATL11.validMask((N_pairs,1), ('data','x_slope','y_slope', 'all','ysearch'))  # 1 col, boolean
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.status=dict()
        self.ref_surf.x_atc=x_atc_ctr
        self.ref_surf.rgt_azimuth=track_azimuth
        self.ref_surf.fit_quality=0
        self.ref_surf.complex_surface_flag=False

    def from_data(self, D11, ind):
        """
        Build an ATL11 point from ATL11 data for a particular point
        """
        D11.index(ind, target=self)
        self.cycles=D11.cycles
        self.N_coeffs=D11.N_coeffs
        self.x_atc_ctr=self.ref_surf.x_atc
        self.y_atc_ctr=self.ref_surf.y_atc
        self.params_11.poly_exponent=D11.poly_exponent
        return self
    
    def select_ATL06_pairs(self, D6, pair_data):
        # Select ATL06 data based on data-quality flags in ATL06 data, and based on
        # consistency checks on the along-track and across-track slope.
        # inputs:
        # D6: ATL06 data structure, containing data from a single RPT as Nx2 arrays
        # pair_data: ATL06_pair structure

        # ATBD section 5.1.2: select "valid pairs" for reference-surface calculation
        # step 1a:  Select segs by data quality
        if not self.ref_surf.complex_surface_flag:            
            self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        else:
            self.valid_segs.data[np.where((D6.SNR_significance<0.01) & (D6.min_along_track_dh < 10) & np.isfinite(D6.h_li))]=True
            
        valid_cycle_count_ATL06_flag=np.unique(D6.cycle_number[np.all(D6.atl06_quality_summary==0, axis=1),:]).size
        min_cycles=np.maximum((np.max(D6.cycle_number) - np.min(D6.cycle_number) +1)/3, 1)
        if (valid_cycle_count_ATL06_flag < min_cycles) or self.ref_surf.complex_surface_flag:
            self.ref_surf.complex_surface_flag=True
            self.valid_segs.data[np.where((D6.snr_significance<0.02) & (D6.min_along_track_dh < 10) & np.isfinite(D6.h_li))]=True
            self.N_cycles_avail=np.unique(D6.cycle_number[np.all(self.valid_segs.data, axis=1),:]).size
        else:
            self.N_cycles_avail=valid_cycle_count_ATL06_flag

        for cc in range(self.cycles[0], self.cycles[1]+1):
            if np.sum(D6.cycle_number==cc) > 0:
                self.cycle_stats.atl06_summary_zero_count[0,cc-self.cycles[0]]=np.sum(self.valid_segs.data[D6.cycle_number==cc])
        self.ref_surf.N_cycle_avail=np.count_nonzero(self.cycle_stats.atl06_summary_zero_count)

        if self.ref_surf.N_cycle_avail<1:
            self.status['No_cycles_available']=True
            return

        # 1b: Select segs by height error
        seg_sigma_threshold=np.maximum(self.params_11.seg_sigma_threshold_min, 3*np.median(D6.h_li_sigma[np.where(self.valid_segs.data)]))
        self.status['N_above_data_quality_threshold']=np.sum(D6.h_li_sigma<seg_sigma_threshold)
        self.valid_segs.data &=  ( D6.h_li_sigma < seg_sigma_threshold )
        self.valid_segs.data &=  np.isfinite(D6.h_li_sigma)

        # 1c: Map valid_segs.data to valid_pairs.data
        self.valid_pairs.data= np.all( self.valid_segs.data, axis=1)
        if not np.any(self.valid_pairs.data):
            self.status['No_valid_pairs_after_height_error_check']=True
            return
        # 2a. see ATL06_pair.py
        # 2b. Calculate the y center of the slope regression
        self.y_polyfit_ctr=np.median(pair_data.y[self.valid_pairs.data])

        # 2c. identify segments close enough to the y center
        self.valid_pairs.ysearch=np.abs(pair_data.y.ravel()-self.y_polyfit_ctr)<self.params_11.L_search_XT

        # 3a: combine data and ysearch
        pairs_valid_for_y_fit= self.valid_pairs.data.ravel() & self.valid_pairs.ysearch.ravel() & np.isfinite(pair_data.dh_dy.ravel())
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

        # calculate the unchanging part of y_slope_tol
        min_my_regression_tol=np.maximum(0.01, 3*np.median(y_slope_sigma))

        # setup the polynomial fit
        my_poly_fit=ATL11.poly_ref_surf(degree_xy=(my_regression_x_degree, my_regression_y_degree), xy0=(self.x_atc_ctr, self.y_polyfit_ctr))
        my_poly_fit.build_fit_matrix(pair_data.x, pair_data.y)
        for iteration in range(2):
            # 3d: regression of across-track slope against pair_data.x and pair_data.y
            y_slope_model, y_slope_resid,  y_slope_chi2r, y_slope_valid_flag=\
                my_poly_fit.fit(pair_data.dh_dy, max_iterations=1, mask=pairs_valid_for_y_fit,  min_sigma=min_my_regression_tol)

            # 3e: calculate across-track slope threshold
            if y_slope_valid_flag.sum()>1:
                y_slope_threshold = np.maximum(min_my_regression_tol,3.*ATL11.RDE(y_slope_resid[y_slope_valid_flag]))
            else:
                y_slope_threshold=min_my_regression_tol
            if ~pairs_valid_for_y_fit.any():
                pairs_valid_for_y_fit=np.zeros_like(pairs_valid_for_y_fit, dtype=bool)
                break
            # 3f: select for across-track residuals within threshold
            pairs_valid_for_y_fit = (np.abs(y_slope_resid)<=y_slope_threshold)
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit_last=pairs_valid_for_y_fit
            pairs_valid_for_y_fit &= self.valid_pairs.data.ravel() & self.valid_pairs.ysearch.ravel()
            if np.all(pairs_valid_for_y_fit == pairs_valid_for_y_fit_last):
                break
        # 3g. Use y slope model to evaluate all pairs
        self.valid_pairs.y_slope=(np.abs(my_poly_fit.z().ravel() - pair_data.dh_dy.ravel()) < y_slope_threshold)[:, np.newaxis]

        #4a. define pairs_valid_for_x_fit
        pairs_valid_for_x_fit= self.valid_pairs.data.ravel() & self.valid_pairs.ysearch.ravel()

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
        # if points are colinear, reduce the y degree to zero
        if len(uY)==2 and len(uX)==2:
            mx_regression_y_degree=0
            
        #4c: Calculate along-track slope regression tolerance
        mx_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:].flatten()))
        x_slope_threshold=0
        mx_poly_fit=ATL11.poly_ref_surf(degree_xy=(mx_regression_x_degree, mx_regression_y_degree), xy0=(self.x_atc_ctr, self.y_polyfit_ctr))
        mx_poly_fit.build_fit_matrix(D6.x_atc.ravel(), D6.y_atc.ravel())

        n_pairs=pair_data.x.size
        for iteration in range(2):
            segs_valid_for_x_fit = np.tile(pairs_valid_for_x_fit[:,np.newaxis], [1,2])

            # 4d: regression of along-track slope against x_pair and y_pair
            if np.sum(pairs_valid_for_x_fit)>0:
                x_slope_model, x_slope_resid,  x_slope_chi2r, _=\
                    mx_poly_fit.fit(D6.dh_fit_dx.ravel(),\
                    mask=segs_valid_for_x_fit.ravel(), max_iterations=1,\
                    min_sigma=mx_regression_tol)

                # 4e: calculate along-track slope threshold
                if x_slope_resid.size > 1.:
                    x_slope_threshold = np.maximum(mx_regression_tol,\
                            3*ATL11.RDE(x_slope_resid[segs_valid_for_x_fit.ravel()]))
                else:
                    x_slope_threshold=mx_regression_tol

                # 4f: select for along-track residuals within threshold
                x_slope_resid.shape=[n_pairs,2]
                self.valid_pairs.x_slope = np.all(x_slope_resid < x_slope_threshold, axis=1)
                pairs_valid_for_x_fit_last=pairs_valid_for_x_fit
                pairs_valid_for_x_fit = self.valid_pairs.data.ravel() & self.valid_pairs.ysearch.ravel() & self.valid_pairs.x_slope.ravel()
                if np.all(pairs_valid_for_x_fit == pairs_valid_for_x_fit_last):
                    break
                if np.sum(pairs_valid_for_x_fit)==0:
                    self.status['no_valid_pairs_for_x_fit']=1
            else:
                self.status['no_valid_pairs_for_x_fit']=1

        # 4g. Use x slope model to evaluate all segments
        self.valid_segs.x_slope=np.abs(mx_poly_fit.z().reshape([n_pairs,2])- D6.dh_fit_dx) < x_slope_threshold #, max_iterations=2, min_sigma=mx_regression_tol)
        self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1)

        # 5: define selected pairs
        self.valid_pairs.all= self.valid_pairs.data.ravel() & self.valid_pairs.y_slope.ravel() & self.valid_pairs.x_slope.ravel()

        if np.sum(self.valid_pairs.all)==0:
            self.status['No_valid_pairs_after_slope_editing']=True

        if np.unique(D6.segment_id[self.valid_pairs.all]).shape[0]==1:
            self.status['Only_one_valid_pair_in_x_direction']=True
        return

    def select_y_center(self, D6, pair_data):  #5.1.3
        # method to select the y_ctr coordinate that allows the maximum number of valid pairs to be included in the fit
        # inputs:
        # D6: ATL06 data structure, containing data from a single RPT as Nx2 arrays
        # pair_data: ATL06_pair structure

        cycle=D6.cycle_number[self.valid_pairs.all,:]
        y_atc=D6.y_atc[self.valid_pairs.all,:]
        # December 10 2015 version:
        y0 = np.nanmedian(np.unique(np.round(pair_data.y)))
        # 1: define a range of y centers, select the center with the best score
        y0_shifts=np.round(y0)+np.arange(-100.5,101.5)
        # 2: search for optimal shift value
        # the score is equal to the number of cycles with at least one valid pair entirely in the window,
        # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
        score=np.zeros_like(y0_shifts)
        y_pair=[]
        for cycle in np.unique(pair_data.cycle[self.valid_pairs.all]):
            y_pair.append(np.nanmedian(pair_data.y[pair_data.cycle==cycle]))
        try:
            N_sel_cycles=np.histogram(y_pair, bins=y0_shifts)[0]
        except ValueError:
            print("Problem with histogram")
        y_unselected=[]
        for cycle in np.unique(D6.cycle_number.ravel()[~np.in1d(D6.cycle_number.ravel(),cycle)]):
            y_unselected.append(np.nanmedian(D6.y_atc[D6.cycle_number==cycle]))
        N_unsel_cycles = np.histogram(y_unselected, bins=y0_shifts)[0]
        count_kernel = np.ones(np.ceil(self.params_11.L_search_XT*2-self.params_11.beam_spacing).astype(int))
        score = np.convolve(N_sel_cycles,count_kernel, mode='same')
        score += np.convolve(N_unsel_cycles, count_kernel, mode='same')/100
        
        # 3: identify the y0_shift value that corresponds to the best score, y_best, formally y_atc_ctr
        best = np.argwhere(score == np.amax(score))
        self.y_atc_ctr=np.median(y0_shifts[best])+0.5*(y0_shifts[1]-y0_shifts[0])
        self.ref_surf.y_atc=self.y_atc_ctr

        if self.DOPLOT is not None and "score-vs-yshift" in self.DOPLOT:
            plt.figure(2);plt.clf()
            plt.plot(y0_shifts,score,'.');
            plt.plot(np.ones_like(np.arange(1,np.amax(score)+1))*self.y_atc_ctr,np.arange(1,np.amax(score)+1),'r')
            plt.title('score vs y0_shifts(blu), y_best(red)')

        # 4: update valid pairs to include y_atc within L_search_XT of y_atc_ctr (y_best)
        self.valid_pairs.ysearch  &= (np.abs(pair_data.y.ravel() - self.y_atc_ctr)<self.params_11.L_search_XT)
        self.valid_pairs.all =  self.valid_pairs.ysearch.ravel() & self.valid_pairs.all.ravel()
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

    def select_fit_columns(self, G_original, selected_segs, deg_wt_sum, TOC):
        
        '''
        Remove design-matrix columns that are uniform (except special cases)
        
        ...also perform a complex-surface check, and force the fit to be 
        overdetermined.
        '''
        # copy the original design matrix
        G=G_original[selected_segs,:]
        # use columns by default
        fit_columns=np.ones(G.shape[1], dtype=bool)
        # 3e. If more than one repeat is present, subset
        # fitting matrix, to include columns that are not uniform
        # 1. remove unused cycles
        all_zero_cycles=np.all(G[:, TOC['zp']]==0, axis=0)
        if np.any(all_zero_cycles):
            fit_columns[TOC['zp'][all_zero_cycles]]=False
        # redundant parameters are correlated with one another
        fit_col_ind=np.flatnonzero(fit_columns)
        Gsq=G[:, fit_columns].T.dot(G[:, fit_columns])
        Gnorm = np.sqrt(np.sum(G[:, fit_columns]**2, axis=0))[None,:]
        Gcorr = Gsq / (Gnorm.T.dot(Gnorm))
        # Gcorr should have ones on the diagonal; any off diagonals that are 1
        # are redundant to the diagonal.  Delete whichever has the larger degree
        redundant_rows, redundant_cols = np.where(np.triu(Gcorr>0.99, k=1))
        if len(redundant_rows) > 0:
            redundant_cols=redundant_cols[ deg_wt_sum[fit_col_ind][redundant_cols] 
                                          > deg_wt_sum[fit_col_ind][redundant_rows] ]
            if len(redundant_cols) > 0:
                fit_columns[fit_col_ind[redundant_cols]]=False
        
        ## 2. check other columns (polynomial and slope-change)
        #columns_to_check=np.setdiff1d(np.arange(G.shape[1], dtype=int), TOC['zp'])
        #columns_to_check=columns_to_check[::-1]
        #for c in columns_to_check:   # check last col first, do in reverse order
        #    if np.max(np.abs(G[:,c]-G[0,c])) < 0.0001:
        #            fit_columns[c]=False
        # if three or more cycle columns are lost, use planar fit in x and y (end of section 3.3)
        if np.sum(np.logical_not(fit_columns[TOC['zp']])) > 2:
            self.ref_surf.complex_surface_flag=True
            # use all segments from the original G_surf
            G=G_original
            selected_segs=np.ones( (np.sum(self.valid_pairs.all)*2),dtype=bool)
            # use only the linear poly columns
            fit_columns[TOC['poly'][self.degree_list_x+self.degree_list_y>1]]=False
        # 3f: Force fit to be overdetermined
        while fit_columns.sum() >= selected_segs.sum():
            # eliminate the column with the largest weighted degree
            fit_columns[deg_wt_sum==deg_wt_sum[fit_columns].max()]=False
        G=G[:, fit_columns]
        return G, fit_columns, selected_segs

    def find_reference_surface(self, D6, pair_data):  #5.1.4
        # method to calculate the reference surface for a reference point
        # Input:
        # D6: ATL06 data structure
 
        # in this section we only consider segments in valid pairs
        self.selected_segments=np.column_stack( (self.valid_pairs.all,self.valid_pairs.all) )
        # Table 4-2
        self.cycle_stats.seg_count=np.zeros((1,self.cycles[1]-self.cycles[0]+1,))

        # establish new boolean arrays for selecting
        selected_pairs=np.ones( (np.sum(self.valid_pairs.all),),dtype=bool)
        selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
        
        # Subset some variables for shorthand
        x_atc      =D6.x_atc[self.valid_pairs.all,:].ravel()
        y_atc      =D6.y_atc[self.valid_pairs.all,:].ravel()
        delta_time =D6.delta_time[self.valid_pairs.all,:].ravel()
        cycle      =D6.cycle_number[self.valid_pairs.all,:].ravel()
        h_li_sigma =D6.h_li_sigma[self.valid_pairs.all,:].ravel()
        h_li       =D6.h_li[self.valid_pairs.all,:].ravel()

        self.ref_surf_cycles = np.unique(cycle)

        # 1. build cycle design matrix with selected segments (those in valid_pairs, initially)
        data=np.ones(len(cycle))
        row=np.array([],dtype=int)
        col=np.array([],dtype=int)
        for index, item in enumerate(self.ref_surf_cycles):
            row=np.append(row,np.nonzero(cycle==item))
            col=np.append(col,np.array(index*np.ones(np.count_nonzero(cycle==item))))
        G_zp=sparse.csc_matrix((data,(row,col)),shape=[len(cycle),len(self.ref_surf_cycles)])

        # 2. determine polynomial degree, using unique x's and unique y's of segments in valid pairs
        # find the maximum number of unique x locations in any cycle
        max_nx_per_cycle=0
        for cycle_i in np.unique(cycle):
            ii = cycle==cycle_i
            max_nx_per_cycle = np.maximum( max_nx_per_cycle, \
                               np.unique(np.round(x_atc[ii]/20).astype(int)).size )
        y_atcU = np.unique(np.round((pair_data.y[self.valid_pairs.all]-self.ref_surf.y_atc)/20).astype(int))
        #np.unique(np.round((y_atc-self.ref_surf.y_atc)/20)) # np.unique orders the unique values
        # Table 4-4   
        self.ref_surf.deg_x = np.maximum(0, np.minimum(self.params_11.poly_max_degree_AT,max_nx_per_cycle-1) )
        self.ref_surf.deg_y = np.maximum(0, np.minimum(self.params_11.poly_max_degree_XT, len(y_atcU)) )
        if self.ref_surf.complex_surface_flag > 0:
            self.ref_surf.deg_x = np.minimum(1, self.ref_surf.deg_x)
            self.ref_surf.deg_y = np.minimum(1, self.ref_surf.deg_y)

        # 3. perform an iterative fit for the across track polynomial
        # 3a. define degree_list_x and degree_list_y.  These are stored in self.default.poly_exponent_list
        degree_x = self.params_11.poly_exponent['x']
        degree_y = self.params_11.poly_exponent['y']
        # keep only degrees > 0 and degree_x+degree_y <= max(max_x_degree, max_y_degree)
        self.poly_mask = (degree_x + degree_y) <= np.maximum(self.ref_surf.deg_x,self.ref_surf.deg_y)
        self.poly_mask &= (degree_x <= self.ref_surf.deg_x)
        self.poly_mask &= (degree_y <= self.ref_surf.deg_y)
        self.degree_list_x = degree_x[self.poly_mask]
        self.degree_list_y = degree_y[self.poly_mask]

        # 3b. define polynomial matrix
        S_fit_poly=ATL11.poly_ref_surf(exp_xy=(self.degree_list_x, self.degree_list_y),\
                                       xy0=(self.x_atc_ctr, self.y_atc_ctr), xy_scale=self.params_11.xy_scale)\
                                        .build_fit_matrix(x_atc, y_atc).fit_matrix

        # 3c. define slope-change matrix
        # TOC is a table-of-contents dict identifying the meaning of the columns of G_surf_zp_original
        TOC=dict()
        TOC['poly']=np.arange(S_fit_poly.shape[1], dtype=int)
        last_poly_col=S_fit_poly.shape[1]-1
        if False: #self.slope_change_t0/self.params_11.t_scale > 1.5/2. and self.ref_surf.deg_x > 0 and self.ref_surf.deg_y > 0:
            self.calc_slope_change=True
            x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_surf_zp_original=np.concatenate( (S_fit_poly,S_fit_slope_change,G_zp.toarray()),axis=1 ) # G = [S St D]
            TOC['slope_change']=last_poly_col+1+np.arange(S_fit_slope_change.shape[1], dtype=int)
            TOC['zp']=TOC['slope_change'][-1]+1+np.arange(G_zp.toarray().shape[1], dtype=int)
        else:
            G_surf_zp_original=np.concatenate( (S_fit_poly,G_zp.toarray()),axis=1 ) # G = [S D]
            TOC['slope_change']=np.array([], dtype=int)
            TOC['zp']=last_poly_col+1+np.arange(G_zp.toarray().shape[1], dtype=int)
        # 3d. build the fitting matrix    
        TOC['surf']=np.concatenate((TOC['poly'], TOC['slope_change']), axis=0)

        # fit_columns is a boolean array identifying those columns of zp_original
        # that survive the fitting process
        fit_columns=np.ones(G_surf_zp_original.shape[1],dtype=bool)
        # (part of 3f) :calculate the order in which the columns will be removed as the size and
        # shape of G changes
        deg_wt_sum=np.zeros_like(fit_columns, dtype=float)
        deg_wt_sum[TOC['poly']]=self.degree_list_x+self.degree_list_y + 0.1*self.degree_list_y
        
        # iterate to remove remaining outlier segments
        for iteration in range(self.params_11.max_fit_iterations):
            #  Make G a copy of G_surf_zp_original, containing only the 
            # selected segs, then:
            
            G, fit_columns, selected_segs = self.select_fit_columns(\
                  G_surf_zp_original, selected_segs, deg_wt_sum, TOC)
            if G.shape[1]==0:
                self.status['inversion failed']=True
                return
            
            # 3g, 3h. generate the data-covariance matrix, its inverse, and
            # the generalized inverse of G
            try:
                C_d, C_di, G_g = gen_inv(self,G,h_li_sigma[selected_segs])
            except:
                self.status['inversion failed'] = True
                return
            # check if any rows of G_g are all-zero (this is in 3h)
            # if so, set the error and return
            if np.any(np.all(G_g==0, axis=1)):
                self.status['inversion failed'] = True
                return

            # inititalize the combined surface and cycle-height model, m_surf_zp
            m_surf_zp=np.zeros(np.size(G_surf_zp_original,1))
            # fill in the columns of m_surf_zp for which we are calculating values
            # the rest are zero
            m_surf_zp[fit_columns]=np.dot(G_g,h_li[selected_segs])

            # 3i. Calculate model residuals for all segments
            r_seg=h_li-np.dot(G_surf_zp_original, m_surf_zp)
            r_fit=r_seg[selected_segs]

            # 3j. Calculate the fitting tolerance,
            r_tol = np.max([1, 3*ATL11.RDE(r_fit/h_li_sigma[selected_segs])])
            # calculate chi-squared value
            misfit_chi2 = np.dot(np.dot(np.transpose(r_fit), C_di),r_fit)

            # calculate P value
            n_cols=np.sum(fit_columns)
            n_rows=np.sum(selected_segs)
            P = 1 - stats.chi2.cdf(misfit_chi2, n_rows-n_cols)

            if self.ref_surf.complex_surface_flag:
                break

            # 3k
            selected_segs_prev=selected_segs.copy()
            if P<0.025 and iteration < self.params_11.max_fit_iterations-1:
                selected_segs = np.abs(r_seg/h_li_sigma) < r_tol # boolean
            
            # make selected_segs pair-wise consistent
            selected_pairs=selected_segs.reshape((len(selected_pairs),2)).all(axis=1)
            selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
            if np.all( selected_segs_prev==selected_segs ):
                break
           
            if not np.any(selected_segs):
                self.status['inversion failed']=True
                return
            if P>0.025:
                break
        if (n_rows-n_cols)>0:
            self.ref_surf.misfit_chi2r=misfit_chi2/(n_rows-n_cols)
        else:
            self.ref_surf.misfit_chi2r=np.NaN
        
        # identify the ref_surf cycles that survived the fit  
        self.ref_surf_cycles=self.ref_surf_cycles[fit_columns[TOC['zp']]]
       
        # map the columns remaining into a TOC that gives the location of each
        # field in the subsetted fitting matrix and a TOC that gives the 
        # destination of each output column
        TOC_sub, TOC_out=remap_TOC(TOC, fit_columns, self.ref_surf_cycles, self.cycles[0])
        self.ref_surf.misfit_RMS = np.sqrt(np.mean(r_fit**2))
        self.selected_segments[np.nonzero(self.selected_segments)] = selected_segs

        # recalculate selected_pairs (possibly redundant, but the break statements can make this necessary)
        selected_pairs = selected_segs.reshape((len(selected_pairs),2)).all(axis=1)
        selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
        #  map selected_pairs (within valid_pairs.all) to all pairs
        selected_pair_out= self.valid_pairs.all.copy()
        selected_pair_out[selected_pair_out==True] = selected_pairs
        # report selected pairs and selected segs
        self.valid_pairs.iterative_fit=selected_pair_out
        self.valid_segs.iterative_fit=np.column_stack((self.valid_pairs.iterative_fit, self.valid_pairs.iterative_fit))

        # write the polynomial components in m_surf_zp to the appropriate columns of
        # self.ref_surf_poly_coeffs
        self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)]=m_surf_zp[TOC['poly']]

        if self.calc_slope_change:
            # the slope change rate columns are scaled as delta_t/t_scale, so they should come out in units of
            # t_scale, or per_year.
            self.ref_surf.slope_change_rate_x= m_surf_zp[TOC['slope_change'][0]]
            self.ref_surf.slope_change_rate_y= m_surf_zp[TOC['slope_change'][1]]
        else:
            self.ref_surf.slope_change_rate_x=np.nan
            self.ref_surf.slope_change_rate_y=np.nan

        # 3l. propagate the errors
        # calculate the data covariance matrix including the scatter component
        h_li_sigma = D6.h_li_sigma[self.selected_segments]
        cycle      = D6.cycle_number[self.selected_segments]
        C_dp = sparse.diags(np.maximum(h_li_sigma**2,(ATL11.RDE(r_fit))**2))
        # calculate the model covariance matrix
        C_m = np.dot(np.dot(G_g,C_dp.toarray()),np.transpose(G_g))
 
        # check for really excessive errors (also in 3l)
        if np.any(np.diagonal(C_m)>1.e4):
            self.status['inversion failed']=True
            return
        # calculate the combined-model errors
        m_surf_zp_sigma=np.zeros_like(m_surf_zp)+np.nan
        m_surf_zp_sigma[fit_columns]=np.sqrt(C_m.diagonal())

        # write out the corrected h values
        cycle_ind=np.zeros(m_surf_zp.shape, dtype=int)-1
        if len(self.ref_surf_cycles) >0:
            cycle_ind[TOC_out['zp']]=self.ref_surf_cycles.astype(int)-self.cycles[0]
        
        # write out the zp
        zp_nan_mask=np.ones_like(TOC_out['zp'], dtype=float)
        zp_nan_mask[m_surf_zp_sigma[TOC_out['zp']]>15]=np.NaN
        self.ROOT.h_corr[0,TOC_out['cycle_ind']]=m_surf_zp[TOC_out['zp']]*zp_nan_mask
       
        # get the square of h_corr_sigma_systematic, equation 12
        sigma_systematic_squared=((D6.dh_fit_dx * D6.sigma_geo_at)**2 + \
            (D6.dh_fit_dy * D6.sigma_geo_xt)**2 + (D6.sigma_geo_r)**2).ravel()

        for ref_cycle in self.ref_surf_cycles.astype(int):
            cc=ref_cycle-self.cycles[0]
            cycle_segs=np.flatnonzero(self.selected_segments)[cycle==ref_cycle]
            W_by_error=h_li_sigma[cycle==ref_cycle]**(-2)/np.sum(h_li_sigma[cycle==ref_cycle]**(-2))

            # weighted means:
            for dataset in ('x_atc','y_atc', 'bsnow_h','r_eff','tide_ocean','dac','h_rms_misfit'): #,'h_rms_misfit'):
                self.cycle_stats.__dict__[dataset][0,cc]=np.sum(W_by_error * getattr(D6, dataset).ravel()[cycle_segs])
            self.cycle_stats.h_mean[0,cc]=np.sum(W_by_error * D6.h_li.ravel()[cycle_segs])

            # root mean weighted square:
            for dataset in ( 'sigma_geo_h','sigma_geo_at','sigma_geo_xt'):
                mean_dataset=dataset #+'_mean';
                self.cycle_stats.__dict__[mean_dataset][0,cc] = np.sqrt(np.sum(W_by_error * getattr(D6, dataset).ravel()[cycle_segs]**2))
            # other parameters:
            self.ROOT.delta_time[0,cc]       = np.mean(D6.delta_time.ravel()[cycle_segs])
            self.cycle_stats.seg_count[0, cc]       = cycle_segs.size
            self.cycle_stats.cloud_flg_asr[0,cc]    = np.min(D6.cloud_flg_asr.ravel()[cycle_segs])
            self.cycle_stats.cloud_flg_atm[0,cc]    = np.min(D6.cloud_flg_atm.ravel()[cycle_segs])
            self.cycle_stats.bsnow_conf[0,cc]       = np.max(D6.bsnow_conf.ravel()[cycle_segs])
            self.cycle_stats.min_snr_significance[0,cc] = np.min(D6.snr_significance.ravel()[cycle_segs])
            self.cycle_stats.min_signal_selection_source[0,cc] = np.min(D6.signal_selection_source.ravel()[cycle_segs])
            if np.isfinite(self.ROOT.h_corr[0,cc]):
                self.ROOT.h_corr_sigma_systematic[0,cc] = np.sqrt(np.sum(W_by_error*sigma_systematic_squared[cycle_segs] ))

        self.ref_surf.N_cycle_used = np.count_nonzero(self.ref_surf_cycles)

        # export the indices of the columns that represent the surface components
        self.surf_mask=TOC_sub['surf']
        # write out the part of the covariance matrix corresponding to the surface model  C_m already corresponds to fit_columns
        self.C_m_surf=C_m[self.surf_mask,:][:,self.surf_mask]
 

        # write out the errors to the data parameters
        self.ref_surf.poly_coeffs_sigma[0,np.where(self.poly_mask)]=m_surf_zp_sigma[TOC['poly']]
        if np.any(self.ref_surf.poly_coeffs_sigma > 2):
            self.status['Polynomial_coefficients_with_high_error']=True
            self.ref_surf.fit_quality += 1

        if self.calc_slope_change:
            self.ref_surf.slope_change_rate_x_sigma=m_surf_zp_sigma[TOC['slope_change'][0]]
            self.ref_surf.slope_change_rate_y_sigma=m_surf_zp_sigma[TOC['slope_change'][1]]
        else:
            self.ref_surf.slope_change_rate_x_sigma= np.nan
            self.ref_surf.slope_change_rate_y_sigma= np.nan

        # write out the errors in h_corr
        self.ROOT.h_corr_sigma[0,TOC_out['cycle_ind']]=m_surf_zp_sigma[TOC_out['zp']]*zp_nan_mask

        if self.DOPLOT is not None and "3D time plot" in self.DOPLOT:
            x_atc = D6.x_atc[self.selected_segments]
            y_atc = D6.y_atc[self.selected_segments]
            x_ctr=np.nanmean(x_atc)
            y_ctr=np.nanmean(y_atc)
            h_li  = D6.h_li[self.selected_segments]
            h_li_sigma = D6.h_li_sigma[self.selected_segments]
            cycle=D6.cycle_number[self.selected_segments]
            fig=plt.figure(31); plt.clf(); ax=fig.add_subplot(111, projection='3d')
            p=ax.scatter(x_atc-x_ctr, y_atc-y_ctr, h_li, c=cycle);
            plt.xlabel('delta x ATC, m')
            plt.ylabel('delta_y ATC, m')
            fig.colorbar(p, label='cycle number')
            fig=plt.figure(32); plt.clf(); ax=fig.add_subplot(111, projection='3d')
            p=ax.scatter(x_atc-x_ctr, y_atc-y_ctr, h_li, c=np.abs(r_seg[selected_segs]/h_li_sigma))
            plt.xlabel('delta x ATC, m')
            plt.ylabel('delta_y ATC, m')
            fig.colorbar(p, label='residual, m')

        return

    def characterize_ref_surf(self):
        """
        method to calculate the slope and curvature of the reference surface
        """
        # make a grid of northing and easting values
        [N,E]=np.meshgrid(np.arange(-50., 60, 10), np.arange(-50., 60, 10))

        # calculate the corresponding values in the ATC system
        xg, yg  = self.local_atc_coords(E, N)

        # evaluate the reference surface at the points in [N,E]
        zg=self.evaluate_reference_surf(xg+self.ref_surf.x_atc, \
                                         yg+self.ref_surf.y_atc, delta_time=None, \
                                         calc_errors=False)

        # fitting a plane as a function of N and E
        G_NE=np.transpose(np.vstack(( (N.ravel()),(E.ravel()), np.ones_like(E.ravel()))))
        msub,rr,rank,sing=linalg.lstsq(G_NE, zg.ravel())

        self.ref_surf.n_slope=msub[0]
        self.ref_surf.e_slope=msub[1]
        self.ref_surf.curvature=np.sqrt(rr)
        if np.any((self.ref_surf.n_slope>0.2,self.ref_surf.e_slope>0.2)):
            self.status['Surface_fit_slope_high']=1
            self.ref_surf.fit_quality += 2

        # perform the same fit in [xg,yg] to calculate the y slope for the unselected segments
        G_xy=np.transpose(np.vstack(( (xg.ravel()),(yg.ravel()), np.ones_like(xg.ravel()))))
        msub_xy, rr, rankxy, singxy=linalg.lstsq(G_xy, zg.ravel())
        self.ref_surf_slope_x=msub_xy[0]
        self.ref_surf_slope_y=msub_xy[1]


    def evaluate_reference_surf(self, x_atc, y_atc, delta_time=None, calc_errors=True):
        """
        method to evaluate the reference surface

        inputs:
            x_atc, y_atc: location to evaluate, in along-track coordinates
            delta_time: time of measurements.  provide delta_time=None to skip the slope-change calculation
            calc_errors: default = true, if set to false, the error calculation is skipped
        """

        poly_mask=np.isfinite(self.ref_surf.poly_coeffs).ravel() 
        x_degree=self.params_11.poly_exponent['x'][poly_mask]
        y_degree=self.params_11.poly_exponent['y'][poly_mask]
        S_fit_poly=ATL11.poly_ref_surf(exp_xy=(x_degree, y_degree),\
                xy0=(self.x_atc_ctr, self.y_atc_ctr), \
                xy_scale=self.params_11.xy_scale).build_fit_matrix(x_atc, y_atc).fit_matrix
        if self.calc_slope_change and (delta_time is not None):
            x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_surf=np.concatenate( (S_fit_poly,S_fit_slope_change),axis=1 ) # G [S St]
            surf_model=np.append(self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)].ravel(),np.array((self.ref_surf.slope_change_rate_x,self.ref_surf.slope_change_rate_y))/self.params_11.t_scale)
        else:
            G_surf=S_fit_poly  #  G=[S]
            surf_model=np.transpose(self.ref_surf.poly_coeffs.ravel()[np.where(poly_mask)])

        G_surf=G_surf[:, self.surf_mask]
        surf_model=surf_model[self.surf_mask]
        # section 3.5
        # calculate corrected heights, z_kc, with non selected segs design matrix and surface shape polynomial from selected segments
        z_ref_surf=np.dot(G_surf,surf_model).ravel()

        if calc_errors is False:
            return z_ref_surf

        # use C_m_surf if it is defined
        if hasattr(self, 'C_m_surf'):
            C_m_surf=self.C_m_surf
        else:
            # if self.C_m_surf is not defined, use the data-product values
            if self.calc_slope_change:
                surf_model_sigma=np.append(self.ref_surf.poly_coeffs_sigma.ravel()[np.where(poly_mask)].ravel(),np.array((self.ref_surf.slope_change_rate_x_sigma,self.ref_surf.slope_change_rate_y_sigma))/self.params_11.t_scale)
            else:
                surf_model_sigma=np.transpose(self.ref_surf.poly_coeffs_sigma.ravel()[np.where(poly_mask)[0]])
            C_m_surf=sparse.diags(surf_model_sigma**2)
        #C_m_surf may be sparse or full -- if sparse, convert to full
        try:
            C_m_surf=C_m_surf.toarray()
        except Exception:
            pass
        z_ref_surf_sigma= np.sqrt( np.diag( np.dot(np.dot(G_surf,C_m_surf),np.transpose(G_surf)) ) ) # equation 11
        return z_ref_surf, z_ref_surf_sigma

    def corr_heights_other_cycles(self, D6):
        # Calculate corrected heights and other parameters for cycles not included in the reference-surface fit
        # input:
        #   D6: ATL06 structure

        # The cycles we are working on are the ones not in ref_surf_cycles
        other_cycles=np.unique(D6.cycle_number.ravel()[~np.in1d(D6.cycle_number.ravel(),self.ref_surf_cycles)])
        # 1. find cycles not in ref_surface_cycles, but have valid_segs.data and valid_segs.x_slope
        non_ref_segments= np.in1d(D6.cycle_number.ravel(),other_cycles) & self.valid_segs.data.ravel() & self.valid_segs.x_slope.ravel()
        #  If the x polynomial degree is zero, allow only segments that have x_atc matching that of the valid segments (+- 10 m)
        if (self.degree_list_x==0).all():
            ref_surf_x_ctrs=D6.x_atc[self.selected_segments]
            ref_surf_x_range=np.array([ref_surf_x_ctrs.min(), ref_surf_x_ctrs.max()])
            non_ref_segments &= (D6.x_atc.ravel() > ref_surf_x_range[0]-10.)
            non_ref_segments &= ( D6.x_atc.ravel() < ref_surf_x_range[1]+10.)

        non_ref_segments &= (np.abs(D6.y_atc.ravel() - self.ref_surf.y_atc) < self.params_11.L_search_XT)

        if ~non_ref_segments.any():
            return
        # select the ATL06 heights and errors from non_ref_segments

        x_atc=D6.x_atc.ravel()[non_ref_segments]
        y_atc=D6.y_atc.ravel()[non_ref_segments]
        delta_time=D6.delta_time.ravel()[non_ref_segments]
        h_li      =D6.h_li.ravel()[non_ref_segments]
        h_li_sigma=D6.h_li_sigma.ravel()[non_ref_segments]
        cycle=D6.cycle_number.ravel()[non_ref_segments]
        #2. build design matrix, G_other, for non selected segments (poly and slope-change parts only)
        z_ref_surf, z_ref_surf_sigma = self.evaluate_reference_surf(x_atc, y_atc, delta_time)
       
        self.non_ref_surf_cycles=np.unique(cycle)
        # section 3.5
        # calculate corrected heights, z_kc, with non selected segs design matrix and surface shape polynomial from selected segments
        z_kc=h_li - z_ref_surf
        z_kc_sigma = np.sqrt( z_ref_surf_sigma**2 + h_li_sigma**2 ) # equation 11

        # get terms of h_corr_sigma_systematic, equation 12
        term1=(D6.dh_fit_dx.ravel()[non_ref_segments] * D6.sigma_geo_at.ravel()[non_ref_segments])**2
        term2=(self.ref_surf_slope_y * D6.sigma_geo_xt.ravel()[non_ref_segments])**2
        term3=(D6.sigma_geo_r.ravel()[non_ref_segments])**2

        non_ref_cycle_ind=np.flatnonzero(non_ref_segments)
        for non_ref_cycle in self.non_ref_surf_cycles.astype(int):
            cc=non_ref_cycle-self.cycles[0]
            # index into the non_ref_segments array:
            best_seg=np.argmin(z_kc_sigma[cycle==non_ref_cycle])
            # index into D6:
            best_seg_ind=non_ref_cycle_ind[cycle==non_ref_cycle][best_seg]
            for dataset in ('x_atc','y_atc','bsnow_h','r_eff','tide_ocean','dac', 'sigma_geo_h','sigma_geo_xt','sigma_geo_at'):
                self.cycle_stats.__dict__[dataset][0,cc]=getattr(D6, dataset).ravel()[best_seg_ind]
            if z_kc_sigma[cycle==non_ref_cycle][best_seg] < 15:
                # edit out errors larger than 15 m                
                self.ROOT.h_corr[0,cc]      =z_kc[cycle==non_ref_cycle][best_seg]
                self.ROOT.h_corr_sigma[0,cc]= z_kc_sigma[cycle==non_ref_cycle][best_seg]
                self.ROOT.h_corr_sigma_systematic[0,cc] = \
                    np.sqrt(term1.ravel()[best_seg] + term2.ravel()[best_seg]  + term3.ravel()[best_seg])
            self.ROOT.delta_time[0,cc]        =D6.delta_time.ravel()[best_seg_ind]
            self.cycle_stats.h_mean[0, cc]         =D6.h_li.ravel()[best_seg_ind]
            self.cycle_stats.min_signal_selection_source[0, cc] = D6.signal_selection_source.ravel()[best_seg_ind]
            self.cycle_stats.min_snr_significance[0, cc] = D6.snr_significance.ravel()[best_seg_ind]
        # establish segment_id_by_cycle for selected segments from reference surface finding and for non_ref_surf
        self.segment_id_by_cycle=[]
        self.selected_segments_by_cycle=[]
        cyc=D6.cycle_number[self.selected_segments[:,0],0]
        segid=D6.segment_id[self.selected_segments[:,0],0]

        non_cyc=D6.cycle_number.ravel()[non_ref_segments]
        non_segid=D6.segment_id.ravel()[non_ref_segments]

        for cc in range(1,D6.cycle_number.max().astype(int)+1):
            if np.in1d(cc,self.ref_surf_cycles):
                self.segment_id_by_cycle.append( np.array( segid[cyc==cc] ) )
            elif np.in1d(cc,self.non_ref_surf_cycles):
                self.segment_id_by_cycle.append( np.array( non_segid[non_cyc==cc] ) )
            else:
                self.segment_id_by_cycle.append(np.array([]))

        self.selected_segments = self.selected_segments | non_ref_segments.reshape(self.valid_pairs.all.shape[0],2)

    def corr_xover_heights(self, D):
        if not isinstance(D, pc.data):
            return
         # find the segments that are within L_search_XT of the reference point
        dE, dN=self.local_NE_coords(D.latitude, D.longitude)
        in_search_radius=(dE**2+dN**2)<self.params_11.L_search_XT**2
        if not np.any(in_search_radius):
            return
        dN=dN[in_search_radius]
        dE=dE[in_search_radius]
        Dsub=D[in_search_radius]
        
        # fix missing Dsub.sigma_geo_r:
        if not hasattr(Dsub, 'sigma_geo_r'):
            Dsub.assign({'sigma_geo_r': np.zeros_like(Dsub.delta_time)+0.03})

        # convert these coordinates in to along-track coordinates
        dx, dy=self.local_atc_coords(dE, dN)

        S_poly=ATL11.poly_ref_surf(exp_xy=(self.degree_list_x, self.degree_list_y),\
            xy0=(0, 0), xy_scale=self.params_11.xy_scale)\
            .build_fit_matrix(dx.ravel(), dy.ravel()).fit_matrix
        surf_model=np.transpose(self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)])
        # pull out the surface-only parts
        S_poly=S_poly[:, self.surf_mask]

        # section 3.5
        # calculate corrected heights, z_xover, and their errors
        z_xover = Dsub.h_li - np.dot(S_poly,surf_model[self.surf_mask]).ravel()
        z_xover_sigma = np.sqrt( np.diag( np.dot(np.dot(S_poly,self.C_m_surf),np.transpose(S_poly)) ) + Dsub.h_li_sigma.ravel()**2 ) # equation 11
        orb_pair=(Dsub.cycle_number-1)*1387+Dsub.rgt+Dsub.BP*0.1
        u_orb_pair=np.unique(orb_pair)
        u_orb_pair=u_orb_pair[~np.in1d(u_orb_pair, self.ref_surf_cycles*1387+self.rgt+self.beam_pair*0.1)]
        ref_surf_slope_mag=np.sqrt(self.ref_surf_slope_x**2+self.ref_surf_slope_y**2)
        for orb_pair_i in u_orb_pair:
            # select the smallest-error segment from each orbit  and pair
            these=np.where(orb_pair==orb_pair_i)[0]
            best=these[np.argmin(z_xover_sigma[these])]

            ss_atc_diff=0
            for di in [-1, 1]:
                this=np.flatnonzero((Dsub.LR[these]==Dsub.LR[best]) & [Dsub.segment_id[these]==Dsub.segment_id[best]+di])
                if len(this)==1:
                    ss_atc_diff += (Dsub.h_li[best]+Dsub.dh_fit_dx[best]*(Dsub.x_atc[best]-Dsub.x_atc[this])-Dsub.h_li[this])**2
            if ss_atc_diff==0:
                ss_atc_diff=[np.NaN]

            # if the along-trac RSS is too large, do not report a value
            if np.sqrt(ss_atc_diff[0]) > 10:
                continue
            
            sigma_systematic = np.sqrt((ref_surf_slope_mag**2 * (Dsub.sigma_geo_xt**2+\
                                                         Dsub.sigma_geo_at**2)) +\
                                                         Dsub.sigma_geo_r**2)

            self.crossing_track_data.rgt.append([Dsub.rgt[best]])
            self.crossing_track_data.spot_crossing.append([Dsub.spot[best]])
            self.crossing_track_data.cycle_number.append([Dsub.cycle_number[best]])
            self.crossing_track_data.h_corr.append([z_xover[best]])
            self.crossing_track_data.h_corr_sigma.append([z_xover_sigma[best]])
            self.crossing_track_data.dac.append([Dsub.dac[best]])
            self.crossing_track_data.tide_ocean.append([Dsub.tide_ocean[best]])
            self.crossing_track_data.delta_time.append([Dsub.delta_time[best]])
            self.crossing_track_data.atl06_quality_summary.append([Dsub.atl06_quality_summary[best]])
            self.crossing_track_data.ref_pt.append([self.ref_pt])
            self.crossing_track_data.latitude.append([self.ROOT.latitude])
            self.crossing_track_data.longitude.append([self.ROOT.longitude])
            self.crossing_track_data.along_track_rss.append([np.sqrt(ss_atc_diff[0])])
            self.crossing_track_data.h_corr_sigma_systematic.append([sigma_systematic[best]])
        return

    def local_NE_coords(self, lat, lon):
        WGS84a=6378137.0
        WGS84b=6356752.31424
        d2r=np.pi/180
        lat0=self.ROOT.latitude
        lon0=self.ROOT.longitude
        Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)
        dE=Re*d2r*(np.mod(lon-lon0+180.,360.)-180.)*np.cos(d2r*lat0)
        dN=Re*d2r*(lat-lat0)
        return dE, dN

    def local_atc_coords(self, dE, dN):
        cos_az=np.cos(self.ref_surf.rgt_azimuth*np.pi/180)
        sin_az=np.sin(self.ref_surf.rgt_azimuth*np.pi/180)
        dx= dN*cos_az + dE*sin_az
        #N.B.  On May 30, I flipped the sign on the dy.  This seems to work better...
        dy= dN*sin_az - dE*cos_az
        return dx, dy

def gen_inv(self,G,sigma):
    # calculate the generalized inverse of matrix G
    # inputs:
    #  G: (NxM) design matrix with one row per data point and one column per parameter
    #  sigma: N-vector of per-data-point errors
    # outputs:
    #  C_d: Data covariance matrix (NxN, sparse)
    #  C_di: Inverse of C_d
    #  G_g: Generalized inverse of G

    # 3g. Generate data-covariance matrix
    #C_d=sparse.diags(sigma**2)
    #C_di=sparse.diags(1/sigma**2)
    #G_sq=np.dot(np.dot(np.transpose(G),C_di.toarray()),G)

    # calculate the generalized inverse of G
    #G_g=np.linalg.solve(G_sq, np.dot(np.transpose(G), C_di.toarray()))
    C_d=np.diag(sigma**2, k=0)
    C_di=np.diag(sigma**-2, k=0)
    G_sq=np.dot(np.dot(np.transpose(G),C_di),G)
    G_g=np.linalg.solve(G_sq, np.dot(np.transpose(G), C_di))
    return C_d, C_di, G_g

def remap_TOC(TOC, fit_columns, ref_surf_cycles, first_cycle):
    """
    Function to handle remapping of columns after subsetting
    """
    TOC_sub={}
    TOC_out={}
    new_cols=np.cumsum(fit_columns)-1
    for field in TOC:
        old_cols=TOC[field][fit_columns[TOC[field]]]
        TOC_sub[field]=new_cols[old_cols]
        TOC_out[field]=old_cols
    # add a field to TOC_out for the cycles
    if len(ref_surf_cycles) >0:
        TOC_out['cycle_ind']=ref_surf_cycles.astype(int)-int(first_cycle)
    else:
        TOC_out['cycle_ind']=None
        
    return TOC_sub, TOC_out
