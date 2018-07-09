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
from ATL11_data import ATL11_data
from ATL11_data import valid_mask
from ATL11_misc import ATL11_defaults
 
class ATL11_point(ATL11_data):
    # ATL11_point is a class with methods for calculating ATL11 from ATL06 data
    def __init__(self, N_pairs=1, ref_pt_number=None, x_atc_ctr=np.NaN,  track_azimuth=np.NaN, max_poly_degree=[1, 1], N_reps=12,  mission_time_bds=None, params_11=None):
        # input variables:
        # N_pairs: Number of distinct pairs in the ATL06 data
        # ref_pt_number: the reference-point number for the ATL11 fit.  This is the geoseg number for the central segment of the fit
        # x_atc_ctr: the along-track corresponding to ref_pt_number
        # track_azimuth: the azimuth of the RGT for the ATL11 point
        # optional parameters:
        # max_poly_degree: the maximum degree of the along- and across-track polynomials
        # N_reps: the number of repeats that might appear in the ATL06 data
        # mission_time_bnds: The start and end of the mission, in delta_time units (seconds)
        # params_11: ATL11_defaults structure

        if params_11 is None:
            self.params_11=ATL11_defaults()
        else:
            self.params_11=params_11
        # initialize the data structure using the ATL11_data __init__ method
        ATL11_data.__init__(self,N_ref_pts=1, N_reps=N_reps, N_coeffs=self.params_11.N_coeffs)
        self.N_pairs=N_pairs
        self.N_reps=N_reps     
        self.N_coeffs=self.params_11.N_coeffs
        self.x_atc_ctr=x_atc_ctr
        self.ref_pt_number=ref_pt_number
        self.track_azimuth=track_azimuth
        self.z_poly_fit=None
        self.mx_poly_fit=None
        self.my_poly_fit=None
        self.ref_surf_slope_x=np.NaN
        self.ref_surf_slope_y=np.NaN
        self.calc_slope_change=False
        
        if mission_time_bds is None:
            mission_time_bds=np.array([0, N_reps*91*24*3600])
        self.slope_change_t0=mission_time_bds[0]+0.5*(mission_time_bds[1]-mission_time_bds[0])
        self.mission_time_bds=mission_time_bds
        self.valid_segs =valid_mask((N_pairs,2), ('data','x_slope','y_slope' ))  #  2 cols, boolan, all F to start
        self.valid_pairs=valid_mask((N_pairs,1), ('data','x_slope','y_slope', 'all','ysearch'))  # 1 col, boolean
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.status=dict()
        self.ref_surf.ref_pt_x_atc=x_atc_ctr
        self.ref_surf.rgt_azimuth=track_azimuth

    def select_ATL06_pairs(self, D6, pair_data):
        # Select ATL06 data based on data-quality flags in ATL06 data, and based on
        # consistency checks on the along-track and across-track slope.
        # inputs:
        # D6: ATL06 data structure, containing data from a single RPT as Nx2 arrays
        # pair_data: ATL06_pair structure
        
        # ATBD section 5.1.2: select "valid pairs" for reference-surface calculation    
        # step 1a:  Select segs by data quality
        self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        self.cycle_stats.ATL06_summary_zero_count=np.zeros((1,self.N_reps,))  # do we need to set to zeros?
        for cc in range(1,self.N_reps+1):
            if D6.cycle[D6.cycle==cc].shape[0] > 0:
                self.cycle_stats.ATL06_summary_zero_count[0,cc-1]=np.sum(self.valid_segs.data[D6.cycle==cc])
                self.cycle_stats.min_SNR_significance[0,cc-1]=np.amin(D6.snr_significance[D6.cycle==cc])
                self.cycle_stats.min_signal_selection_source[0,cc-1]=np.amin(D6.signal_selection_source[D6.cycle==cc])
        self.ref_surf.N_cycle_avail=np.count_nonzero(self.cycle_stats.ATL06_summary_zero_count) 
        
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
            # 3d: regression of across-track slope against pair_data.x and pair_data.y
            self.my_poly_fit=poly_ref_surf(degree_xy=(my_regression_x_degree, my_regression_y_degree), xy0=(self.x_atc_ctr, self.y_polyfit_ctr)) 
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
            self.mx_poly_fit=poly_ref_surf(degree_xy=(mx_regression_x_degree, mx_regression_y_degree), xy0=(self.x_atc_ctr, self.y_polyfit_ctr)) 
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
        # method to select the y_ctr coordinate that allows the maximum number of valid pairs to be included in the fit
        # inputs:
        # D6: ATL06 data structure, containing data from a single RPT as Nx2 arrays
        # pair_data: ATL06_pair structure

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
            
            other_cycles=np.unique(cycle.ravel()[~np.in1d(cycle.ravel(),sel_cycs)])
            unsel_segs=np.logical_and(np.in1d(cycle.ravel(),other_cycles),np.abs(D6.y_atc[self.valid_pairs.all,:].ravel()-y0_shift)<self.params_11.L_search_XT)
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
        
        # 4: update valid pairs to include y_atc within L_search_XT of y_atc_ctr (y_best)
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

    def find_reference_surface(self, D6):  #5.1.4
        # method to calculate the reference surface for a reference point
        # Input:
        # D6: ATL06 data structure
        self.corrected_h.quality_summary=np.logical_not(np.logical_and( np.logical_and(self.cycle_stats.min_signal_selection_source<=1, self.cycle_stats.min_SNR_significance<0.02),self.cycle_stats.ATL06_summary_zero_count>0 ))        
        self.ref_surf.complex_surface_flag=0
        self.ref_surf.surf_fit_quality_summary=0
        
        # in this section we only consider segments in valid pairs
        self.selected_segments=np.column_stack( (self.valid_pairs.all,self.valid_pairs.all) )
        # Table 4-2        
        self.cycle_stats.cycle_seg_count=np.zeros((1,self.N_reps,))
        self.cycle_stats.cycle_included_in_fit=np.zeros((1,self.N_reps,))

        # establish new boolean arrays for selecting
        selected_pairs=np.ones( (np.sum(self.valid_pairs.all),),dtype=bool) 
        selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()  

        cycle=D6.cycle[self.valid_pairs.all,:].ravel()  # want the cycle of each seg in valid pair
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
        #print(self.degree_list_x,self.degree_list_x+self.degree_list_y)
        # 3b. define polynomial matrix
        x_atc=D6.x_atc[self.valid_pairs.all,:].ravel()
        y_atc=D6.y_atc[self.valid_pairs.all,:].ravel()
        S_fit_poly=poly_ref_surf(exp_xy=(self.degree_list_x, self.degree_list_y), xy0=(self.x_atc_ctr, self.y_atc_ctr), xy_scale=self.params_11.xy_scale).fit_matrix(x_atc, y_atc)
        
        # 3c. define slope-change matrix 
        # 3d. build the fitting matrix
        delta_time=D6.delta_time[self.valid_pairs.all,:].ravel()

        # TOC is a table-of-contents dict identifying the meaning of the columns 
        # of G_surf_zp_original
        TOC=dict()
        if self.slope_change_t0/self.params_11.t_scale > 1.5/2.:
            self.calc_slope_change=True
            x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_surf_zp_original=np.concatenate( (S_fit_poly,S_fit_slope_change,G_zp.toarray()),axis=1 ) # G = [S St D]
            TOC['poly']=np.arange(S_fit_poly.shape[1])
            TOC['slope_change']=TOC['poly'][-1]+1+np.arange(S_fit_slope_change.shape[1])
            TOC['zp']=TOC['slope_change'][-1]+1+np.arange(G_zp.toarray().shape[1])
        else:
            G_surf_zp_original=np.concatenate( (S_fit_poly,G_zp.toarray()),axis=1 ) # G = [S D]
            TOC['poly']=np.arange(S_fit_poly.shape[1])
            TOC['slope_change']=np.array([])
            TOC['zp']=TOC['poly'][-1]+1+np.arange(G_zp.toarray().shape[1])
        TOC['surf']=np.concatenate((TOC['poly'], TOC['slope_change']), axis=0)
        
        # fit_columns is a boolean array identifying those columns of zp_original 
        # that survive the fitting process
        fit_columns=np.ones(G_surf_zp_original.shape[1],dtype=bool)

        # reduce these variables to the segments of the valid pairs only
        h_li_sigma=D6.h_li_sigma[self.valid_pairs.all,:].ravel() 
        h_li      =D6.h_li[self.valid_pairs.all,:].ravel()
            
        for kk in range(self.params_11.max_fit_iterations): 
            #  Make G a copy of G_surf_zp_original, containing only the selected segs
            G=G_surf_zp_original[selected_segs,:]
            # 3e. If more than one repeat is present, subset 
            #fitting matrix, to include columns that are not uniform
            if self.ref_surf_cycles.size > 1:
                columns_to_check=range(G.shape[1]-1,-1,-1)
            else:
                #Otherwise, check only the surface columns
                columns_to_check=range(TOC['zp'][0]-1, -1, -1)
                self.ref_surf.surf_fit_quality_summary=1
            for c in columns_to_check:   # check last col first, do in reverse order
                if np.max(np.abs(G[:,c]-G[0,c])) < 0.0001:
                        fit_columns[c]=False
            # if three or more cycle columns are lost, use planar fit in x and y (end of section 3.3)
            if np.sum(np.logical_not(fit_columns[TOC['zp']])) > 2:
                self.ref_surf.complex_surface_flag=1
                # use all segments from the original G_surf     
                G=G_surf_zp_original
                selected_segs=np.ones( (np.sum(self.valid_pairs.all)*2),dtype=bool)      
                # use only the linear poly columns
                fit_columns[TOC['poly'][self.degree_list_x+self.degree_list_y>1]]=False
            G=G[:, fit_columns]
            if G.shape[0] < G.shape[1]:
                self.status['inversion failed']=True
                return

            # 3f, 3g. generate the data-covariance matrix, its inverse, and 
            # the generalized inverse of G
            C_d, C_di, G_g = gen_inv(self,G,h_li_sigma[selected_segs])
                        
            # inititalize the combined surface and cycle-height model, m_surf_zp 
            m_surf_zp=np.zeros(np.size(G_surf_zp_original,1))            
            z=h_li[selected_segs]
            # fill in the columns of m_surf_zp for which we are calculating values
            # the rest are zero            
            m_surf_zp[fit_columns]=np.dot(G_g,z)  

            # 3h. Calculate model residuals for all segments
            r_seg=h_li-np.dot(G_surf_zp_original,m_surf_zp)            
            r_fit=r_seg[selected_segs] 
            
            # 3i. Calculate the fitting tolerance, 
            r_tol = 3*RDE(r_fit/h_li_sigma[selected_segs])
            # reduce chi-squared value
            surf_fit_misfit_chi2 = np.dot(np.dot(np.transpose(r_fit),C_di.toarray()),r_fit)

            # calculate P value
            n_cols=np.sum(fit_columns)
            n_rows=np.sum(selected_segs)
            P = 1 - stats.chi2.cdf(surf_fit_misfit_chi2, n_rows-n_cols)
            
            if self.ref_surf.complex_surface_flag==1:
                break
            
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
        if (n_rows-n_cols)>0:
            self.ref_surf.surf_fit_misfit_chi2r=surf_fit_misfit_chi2/(n_rows-n_cols)
        else:
            self.ref_surf.surf_fit_misfit_chi2r=np.NaN
        self.ref_surf.surf_fit_misfit_RMS=RDE(r_fit)   # Robust Dispersion Estimate, half the diff bet the 16th and 84th percentiles of a distribution        
        self.selected_segments[np.nonzero(self.selected_segments)]=selected_segs #??? should this be valid.iterative_fit???
        # identify the ref_surf cycles that survived the fit         
        self.ref_surf_cycles=self.ref_surf_cycles[fit_columns[TOC['zp']]]            
        
        # report the selected segments ####BEN_FIX_THIS???
        selected_pair_out= self.valid_pairs.all.copy()
        selected_pair_out[selected_pair_out==True]=selected_segs.reshape((len(selected_pairs),2)).all(axis=1)
        
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

        # check if slope change rate for either x or y is > 0.1 (see Table 4-4)     
        if np.any((np.logical_or(np.abs(self.ref_surf.slope_change_rate_x)>0.1,np.isnan(self.ref_surf.slope_change_rate_x)),
                   np.logical_or(np.abs(self.ref_surf.slope_change_rate_y)>0.1,np.isnan(self.ref_surf.slope_change_rate_y)))):
            self.ref_surf.surf_fit_quality_summary=1
        
        # write out the corrected h values
        cycle_ind=np.zeros(m_surf_zp.shape, dtype=int)-1
        cycle_ind[TOC['zp']]=np.arange(TOC['zp'].size, dtype=int)
        zp_mask=TOC['zp'][fit_columns[TOC['zp']]]        
        self.corrected_h.cycle_h_shapecorr[0,cycle_ind[zp_mask]]=m_surf_zp[zp_mask]
      
        # get the square of cycle_h_shapecorr_sigma_systematic, equation 12
        sigma_systematic_squared=((D6.dh_fit_dx * D6.sigma_geo_at)**2 + \
            (D6.dh_fit_dy * D6.sigma_geo_xt)**2 + (D6.sigma_geo_h)**2).ravel() 

        h_li_sigma = D6.h_li_sigma[self.selected_segments]
        cycle      = D6.cycle[self.selected_segments]
        for cc in self.ref_surf_cycles.astype(int):
            cycle_segs=np.flatnonzero(self.selected_segments)[cycle==cc]
            W_by_error=h_li_sigma[cycle==cc]**(-2)/np.sum(h_li_sigma[cycle==cc]**(-2))

            # weighted means:
            for dataset in ('latitude','longitude','x_atc','y_atc', 'bsnow_h','r_eff','tide_ocean','h_robust_spread','h_rms_misft'):
                mean_dataset=dataset+'_mean';
                self.cycle_stats.__dict__[mean_dataset][0,cc-1]=np.sum(W_by_error * getattr(D6, dataset).ravel()[cycle_segs])
            self.cycle_stats.h_uncorr_mean[0,cc-1]=np.sum(W_by_error * D6.h_li.ravel()[cycle_segs]) 
            
            # root mean weighted square:
            for dataset in ( 'sigma_geo_h','sigma_geo_at','sigma_geo_xt'):
                mean_dataset=dataset+'_mean';
                self.cycle_stats.__dict__[mean_dataset][0,cc-1]=np.sqrt(np.sum(W_by_error * getattr(D6, dataset).ravel()[cycle_segs]**2))
            # other parameters:     
            self.corrected_h.mean_cycle_time[0,cc-1]       =np.mean(D6.delta_time.ravel()[cycle_segs])            
            self.cycle_stats.cycle_included_in_fit[0,cc-1]=1
            self.cycle_stats.cycle_seg_count[0, cc-1]=cycle_segs.size            
            self.cycle_stats.cloud_flg_asr_best[0,cc-1]    =np.min(D6.cloud_flg_asr.ravel()[cycle_segs])
            self.cycle_stats.cloud_flg_atm_best[0,cc-1]    =np.min(D6.cloud_flg_atm.ravel()[cycle_segs])
            self.cycle_stats.bsnow_conf_best[0,cc-1]       =np.max(D6.bsnow_conf.ravel()[cycle_segs])
            self.corrected_h.cycle_h_shapecorr_sigma_systematic[0,cc-1] =np.sqrt(np.sum(W_by_error*sigma_systematic_squared[cycle_segs] ))

        self.ref_surf.N_cycle_used=np.count_nonzero(self.ref_surf_cycles)
        if self.ref_surf.N_cycle_used<2:
            self.ref_surf.surf_fit_quality_summary=1
        
        # 3k. propagate the errors
        # calculate the data covariance matrix including the scatter component
        C_dp=sparse.diags(np.maximum(h_li_sigma**2,(RDE(r_fit))**2))  
        # calculate the model covariance matrix
        C_m = np.dot(np.dot(G_g,C_dp.toarray()),np.transpose(G_g))
        # calculate the combined-model errors
        m_surf_zp_sigma=np.zeros_like(m_surf_zp)+np.nan
        m_surf_zp_sigma[fit_columns]=np.sqrt(C_m.diagonal())
        
        # identify which of the columns that were included in the fit belong to the surface model
        surf_mask=np.arange(np.sum(fit_columns[TOC['surf']]))
        # write out the part of the covariance matrix corresponding to the surface model   
        self.C_m_surf=C_m[surf_mask,:][:,surf_mask]
        # export the indices of the columns that represent the surface components
        self.surf_mask=np.flatnonzero(fit_columns[TOC['surf']])
        

        # write out the errors to the data parameters
        self.ref_surf.poly_coeffs_sigma[0,np.where(self.poly_mask)]=m_surf_zp_sigma[TOC['poly']]
        if self.calc_slope_change:          
            self.ref_surf.slope_change_rate_x_sigma=m_surf_zp_sigma[ TOC['slope_change'][0]] 
            self.ref_surf.slope_change_rate_y_sigma=m_surf_zp_sigma[ TOC['slope_change'][1]] 
        else:
            self.ref_surf.slope_change_rate_x_sigma= np.nan
            self.ref_surf.slope_change_rate_y_sigma= np.nan
        
        # write out the errors in h_shapecorr
        self.corrected_h.cycle_h_shapecorr_sigma[0,cycle_ind[zp_mask]]=m_surf_zp_sigma[zp_mask]
          
        # calculate fit slopes and curvature:
        # make a grid of northing and easting values
        [N,E]=np.meshgrid(np.arange(-50., 60, 10),np.arange(-50., 60, 10))
        
        # calculate the corresponding values in the ATC system
        cos_az=np.cos(self.ref_surf.rgt_azimuth*np.pi/180) 
        sin_az=np.sin(self.ref_surf.rgt_azimuth*np.pi/180)         
        xg= N*cos_az + E*sin_az           
        yg=-N*sin_az + E*cos_az
           
        zg=np.zeros_like(xg)
        for ii in np.arange(np.sum(self.poly_mask)):
            xterm=( xg/self.params_11.xy_scale )**self.degree_list_x[ii]
            yterm=( yg/self.params_11.xy_scale )**self.degree_list_y[ii]
            zg=zg+self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)][0,ii] * xterm * yterm 
   
        # fitting a plane as a function of N and E 
        G_NE=np.transpose(np.vstack(( (N.ravel()),(E.ravel()), np.ones_like(E.ravel()))))
        msub,rr,rank,sing=linalg.lstsq(G_NE, zg.ravel())
        
        self.ref_surf.fit_N_slope=msub[0] 
        self.ref_surf.fit_E_slope=msub[1] 
        self.ref_surf.fit_curvature=np.sqrt(rr)

        # perform the same fit in [xg,yg] to calculate the y slope for the unselected segments
        G_xy=np.transpose(np.vstack(( (xg.ravel()),(yg.ravel()), np.ones_like(xg.ravel()))))
        msub_xy, rr, rankxy, singxy=linalg.lstsq(G_xy, zg.ravel())
        self.ref_surf_slope_x=msub_xy[0]
        self.ref_surf_slope_y=msub_xy[1]

        if self.DOPLOT is not None and "3D time plot" in self.DOPLOT:
            x_atc = D6.x_atc[self.selected_segments]       
            y_atc = D6.y_atc[self.selected_segments]
            x_ctr=np.nanmean(x_atc)
            y_ctr=np.nanmean(y_atc)
            h_li  = D6.h_li[self.selected_segments]        
            h_li_sigma = D6.h_li_sigma[self.selected_segments]
            cycle=D6.cycle[self.selected_segments]
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
    
    def corr_heights_other_cycles(self, D6):
        # Calculate corrected heights and other parameters for cycles not included in the reference-surface fit
        # input:
        #   D6: ATL06 structure

        # The cycles we are working on are the ones not in ref_surf_cycles
        other_cycles=np.unique(D6.cycle.ravel()[~np.in1d(D6.cycle.ravel(),self.ref_surf_cycles)])
        # 1. find cycles not in ref_surface_cycles, but have valid_segs.data and valid_segs.x_slope  
        non_ref_segments=np.logical_and(np.in1d(D6.cycle.ravel(),other_cycles),np.logical_and(self.valid_segs.data.ravel(),self.valid_segs.x_slope.ravel()))
        #  If the x polynomial degree is zero, allow only segments that have x_atc matching that of the valid segments (+- 10 m)
        if (self.degree_list_x==0).all():
            ref_surf_x_ctrs=D6.x_atc[self.selected_segments]
            ref_surf_x_range=np.array([ref_surf_x_ctrs.min(), ref_surf_x_ctrs.max()])
            non_ref_segments=np.logical_and(non_ref_segments, D6.x_atc.ravel() > ref_surf_x_range[0]-10.)
            non_ref_segments=np.logical_and(non_ref_segments, D6.x_atc.ravel() < ref_surf_x_range[1]+10.)
        
        if ~non_ref_segments.any():
            return
        # 2. build design matrix, G_other, for non selected segments (poly and slope-change parts only)
        x_atc=D6.x_atc.ravel()[non_ref_segments]
        y_atc=D6.y_atc.ravel()[non_ref_segments]
        S_fit_poly=poly_ref_surf(exp_xy=(self.degree_list_x, self.degree_list_y), xy0=(self.x_atc_ctr, self.y_atc_ctr), xy_scale=self.params_11.xy_scale).fit_matrix(x_atc, y_atc)

        if self.calc_slope_change:
            delta_time=D6.delta_time.ravel()[non_ref_segments]
            x_term=np.array( [(x_atc-self.x_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/self.params_11.xy_scale * (delta_time-self.slope_change_t0)/self.params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_other=np.concatenate( (S_fit_poly,S_fit_slope_change),axis=1 ) # G [S St]
            surf_model=np.append(self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)].ravel(),np.array((self.ref_surf.slope_change_rate_x,self.ref_surf.slope_change_rate_y))/self.params_11.t_scale)
        else:
            G_other=S_fit_poly  #  G=[S]
            surf_model=np.transpose(self.ref_surf.poly_coeffs[0,np.where(self.poly_mask)])    
        # pull out the surface-only parts
        G_other=G_other[:, self.surf_mask]
        # select the ATL06 heights and errors from non_ref_segments 
        h_li      =D6.h_li.ravel()[non_ref_segments]
        h_li_sigma=D6.h_li_sigma.ravel()[non_ref_segments]
        cycle=D6.cycle.ravel()[non_ref_segments]
        
        self.non_ref_surf_cycles=np.unique(cycle)
        # section 3.5
        # calculate corrected heights, z_kc, with non selected segs design matrix and surface shape polynomial from selected segments
        z_kc=h_li - np.dot(G_other,surf_model[self.surf_mask]).ravel()         
        z_kc_sigma = np.sqrt( np.diag( np.dot(np.dot(G_other,self.C_m_surf),np.transpose(G_other)) ) + h_li_sigma**2 ) # equation 11
         
        # get terms of cycle_h_shapecorr_sigma_systematic, equation 12
        term1=(D6.dh_fit_dx * D6.sigma_geo_at)**2
        term2=(self.ref_surf_slope_y * D6.sigma_geo_xt)**2
        term3=(D6.sigma_geo_h)**2
        
        for cc in self.non_ref_surf_cycles.astype(int):
            best_seg=np.argmin(z_kc_sigma[cycle==cc])
            best_seg_ind=np.flatnonzero(non_ref_segments)[cycle==cc][best_seg]
            for dataset in ('latitude','longitude','x_atc','y_atc','bsnow_h','r_eff','tide_ocean','h_robust_spread','sigma_geo_h','sigma_geo_xt','sigma_geo_at'):
                mean_dataset=dataset+'_mean';
                self.cycle_stats.__dict__[mean_dataset][0,cc-1]=getattr(D6, dataset).ravel()[best_seg_ind]
            self.corrected_h.cycle_h_shapecorr[0,cc-1]      =z_kc[cycle==cc][best_seg]
            self.corrected_h.cycle_h_shapecorr_sigma[0,cc-1]= z_kc_sigma[cycle==cc][best_seg]
            self.corrected_h.cycle_h_shapecorr_sigma_systematic[0,cc-1] = \
                np.sqrt(term1.ravel()[best_seg_ind] + term2.ravel()[best_seg_ind]  + term3.ravel()[best_seg_ind])
            self.corrected_h.mean_cycle_time[0,cc-1]        =D6.delta_time.ravel()[best_seg_ind]
            self.cycle_stats.cycle_seg_count[0,cc-1]         =1
            self.cycle_stats.h_uncorr_mean[0, cc-1]         =D6.h_li.ravel()[best_seg_ind]
        # establish segment_id_by_cycle for selected segments from reference surface finding and for non_ref_surf
        self.segment_id_by_cycle=[]         
        self.selected_segments_by_cycle=[]         
        cyc=D6.cycle[self.selected_segments[:,0],0]  
        segid=D6.segment_id[self.selected_segments[:,0],0]    
    
        non_cyc=D6.cycle.ravel()[non_ref_segments]  
        non_segid=D6.segment_id.ravel()[non_ref_segments]  
    
        for cc in range(1,D6.cycle.max().astype(int)+1):  
            if np.in1d(cc,self.ref_surf_cycles):
                self.segment_id_by_cycle.append( np.array( segid[cyc==cc] ) )
            elif np.in1d(cc,self.non_ref_surf_cycles):
                self.segment_id_by_cycle.append( np.array( non_segid[non_cyc==cc] ) )
            else:     
                self.segment_id_by_cycle.append(np.array([]))

        self.selected_segments=np.logical_or(self.selected_segments,non_ref_segments.reshape(self.valid_pairs.all.shape[0],2))


def gen_inv(self,G,sigma):
    # calculate the generalized inverse of matrix G
    # inputs:
    #  G: (NxM) design matrix with one row per data point and one column per parameter
    #  sigma: N-vector of per-data-point errors
    # outputs:
    #  C_d: Data covariance matrix (NxN, sparse)
    #  C_di: Inverse of C_d
    #  G_g: Generalized inverse of G

    # 3f. Generate data-covariance matrix
    C_d=sparse.diags(sigma**2)
    C_di=sparse.diags(1/sigma**2)
    G_sq=np.dot(np.dot(np.transpose(G),C_di.toarray()),G)
    G_sqi=linalg.inv(G_sq)
    # calculate the generalized inverse of G
    G_g=np.dot( np.dot(G_sqi,np.transpose(G)),C_di.toarray() )
    return C_d, C_di, G_g
