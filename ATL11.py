# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017

@author: ben
"""

import numpy as np
from poly_ref_surf import poly_ref_surf
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from RDE import RDE
import scipy.sparse as sparse
from scipy import linalg 
from scipy import stats
import scipy.sparse.linalg as sps_linalg
import time

class generic_group:
    def __init__(self, N_ref_pts, N_reps, per_pt_fields, full_fields):
        for field in per_pt_fields:
            setattr(self, field, np.zeros([N_ref_pts, 1]))
        for field in full_fields:
            setattr(self, field, np.zeros([N_ref_pts, N_reps]))
        self.per_pt_fields=per_pt_fields
        self.full_fields=full_fields
        self.list_of_fields=self.per_pt_fields.append(self.full_fields)
               
class valid_mask:
    def __init__(self, dims, fields):
        for field in fields:
            setattr(self, field, np.zeros(dims, dtype='bool'))

class ATL11_data:
    def __init__(self, N_ref_pts, N_reps):
        self.Data=[]
        self.DOPLOT=None
        # define empty records here based on ATL11 ATBD
        self.corrected_h=generic_group(N_ref_pts, N_reps, ['ref_pt_lat', 'ref_pt_lon', 'ref_pt_number'], ['mean_pass_time', 'pass_h_shapecorr', 'pass_h_shapecorr_sigma','pass_h_shapecorr_sigma_systematic','quality_summary'])
        
class ATL11_point:
    def __init__(self, N_pairs=1, x_atc_ctr=np.NaN,  y_atc_ctr=np.NaN, track_azimuth=np.NaN, max_poly_degree=[1, 1], N_reps=12):
        self.x_atc_ctr=x_atc_ctr
        self.y_atc_ctr=y_atc_ctr
        self.z_poly_fit=None
        self.mx_poly_fit=None
        self.my_poly_fit=None
        self.valid_segs =valid_mask((N_pairs,2),  ('data','x_slope','y_slope' ))  #  2 cols, boolan, all F to start
        self.valid_pairs=valid_mask((N_pairs,1), ('data','x_slope','y_slope', 'all','ysearch'))  # 1 col, boolean
        self.unselected_cycle_segs=np.zeros((N_pairs,2), dtype='bool')
        self.z=ATL11_data(1, N_reps)
        self.status=dict()
        self.DOPLOT=None

    def select_ATL06_pairs(self, D6, pair_data, params_11):   # x_polyfit_ctr is x_atc_ctr and seg_x_center
        # this is section 5.1.2: select "valid pairs" for reference-surface calculation    
        # step 1a:  Select segs by data quality
        self.valid_segs.data[np.where(D6.atl06_quality_summary==0)]=True
        # step 1b; the backup step here is UNDOCUMENTED AND UNTESTED
        if not np.any(self.valid_segs.data):
            self.status['atl06_quality_summary_all_nonzero']=1.0
            self.valid_segs.data[np.where(np.logical_or(D6.snr_significance<0.02, D6.signal_selection_source <=2))]=True
            if not np.any(self.valid_segs.data):
                self.status['atl06_quality_all_bad']=1
                return 
        # 1b: Select segs by height error        
        seg_sigma_threshold=np.maximum(params_11.seg_sigma_threshold_min, 3*np.median(D6.h_li_sigma[np.where(self.valid_segs.data)]))
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
        self.valid_pairs.ysearch=np.abs(pair_data.y.ravel()-self.y_polyfit_ctr)<params_11.L_search_XT  
        
        # 3a: combine data and ysearch
        pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel()) 
        # 3b:choose the degree of the regression for across-track slope
        if len(np.unique(pair_data.x[pairs_valid_for_y_fit]))>1:
            my_regression_x_degree=1
        else:
            my_regression_x_degree=0
        if len(np.unique(pair_data.y[pairs_valid_for_y_fit]))>1:
            my_regression_y_degree=1
        else:
            my_regression_y_degree=0
    
        # 3c: Calculate across-track slope regression tolerance
        y_slope_sigma=np.sqrt(np.sum(D6.h_li_sigma[pairs_valid_for_y_fit,:]**2, axis=1))/np.transpose(np.diff(D6.y_atc[pairs_valid_for_y_fit,:], axis=1)).ravel() #same shape as y_slope*
        my_regression_tol=np.max(0.01, 3*np.median(y_slope_sigma))

        for item in range(2):
            # 3d: regression of across-track slope against pair_data.x and pair_data.y
            self.my_poly_fit=poly_ref_surf(my_regression_x_degree, my_regression_y_degree, self.x_atc_ctr, self.y_polyfit_ctr) 
            y_slope_model, y_slope_resid,  y_slope_chi2r, y_slope_valid_flag=self.my_poly_fit.fit(pair_data.x[pairs_valid_for_y_fit], pair_data.y[pairs_valid_for_y_fit], D6.dh_fit_dy[pairs_valid_for_y_fit,0], max_iterations=2, min_sigma=my_regression_tol)
            # update what is valid based on regression flag
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=y_slope_valid_flag                #re-establish pairs_valid for y fit
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.y_slope.ravel()) # what about ysearch?
            
            # 3e: calculate across-track slope threshold
            y_slope_threshold=np.max(my_regression_tol,3*RDE(y_slope_resid))
            
            # 3f: select for across-track residuals within threshold
            self.valid_pairs.y_slope[np.where(pairs_valid_for_y_fit),0]=np.abs(y_slope_resid)<=y_slope_threshold
            # re-establish pairs_valid_for_y_fit
            pairs_valid_for_y_fit=np.logical_and( np.logical_and(self.valid_pairs.data.ravel(),self.valid_pairs.ysearch.ravel()), self.valid_pairs.y_slope.ravel()) 
                            
        # 3g. Use model on all pairs
        self.valid_pairs.y_slope=np.abs(self.my_poly_fit.z(pair_data.x, pair_data.y)- pair_data.dh_dy) < y_slope_threshold 
        
        #4a. define pairs_valid_for_x_fit
        pairs_valid_for_x_fit= np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.ysearch.ravel())
        # 4b:choose the degree of the regression for along-track slope
        if len(np.unique(D6.x_atc[pairs_valid_for_x_fit,:].ravel()))>1:
            mx_regression_x_degree=1
        else:
            mx_regression_x_degree=0
        if len(np.unique(D6.y_atc[pairs_valid_for_x_fit,:].ravel()))>1:
            mx_regression_y_degree=1
        else:
            mx_regression_y_degree=0

        #4c: Calculate along-track slope regression tolerance
        mx_regression_tol=np.maximum(0.01, 3*np.median(D6.dh_fit_dx_sigma[pairs_valid_for_x_fit,:].flatten())) 
        for item in range(2):
            # 4d: regression of along-track slope against x_pair and y_pair
            self.mx_poly_fit=poly_ref_surf(mx_regression_x_degree, mx_regression_y_degree, self.x_atc_ctr, self.y_polyfit_ctr) 
            x_slope_model, x_slope_resid,  x_slope_chi2r, x_slope_valid_flag=self.mx_poly_fit.fit(D6.x_atc[pairs_valid_for_x_fit,:].ravel(), D6.y_atc[pairs_valid_for_x_fit,:].ravel(), D6.dh_fit_dx[pairs_valid_for_x_fit,:].ravel(), max_iterations=2, min_sigma=mx_regression_tol)
            # update what is valid based on regression flag
            x_slope_valid_flag.shape=[np.sum(pairs_valid_for_x_fit),2]
            self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=x_slope_valid_flag
            self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
            # re-establish pairs_valid_for_y_fit
            # re-establish pairs_valid_for_x_fit
            pairs_valid_for_x_fit=np.logical_and(self.valid_pairs.data.ravel(), self.valid_pairs.x_slope.ravel()) # include ysearch here?
            
            # 4e: calculate along-track slope threshold
            x_slope_threshold = np.max(mx_regression_tol,3*RDE(x_slope_resid))

            # 4f: select for along-track residuals within threshold
            x_slope_resid.shape=[np.sum(pairs_valid_for_x_fit),2]
            self.valid_segs.x_slope[np.where(pairs_valid_for_x_fit),:]=np.transpose(np.tile(np.all(np.abs(x_slope_resid)<=x_slope_threshold,axis=1),(2,1)))
            self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
            pairs_valid_for_x_fit=np.logical_and(np.logical_and(self.valid_pairs.data.ravel(),self.valid_pairs.ysearch.ravel()), self.valid_pairs.x_slope.ravel()) 
            
        # 4g. Use model on all segments
        self.valid_segs.x_slope=np.abs(self.mx_poly_fit.z(D6.x_atc, D6.y_atc)- D6.dh_fit_dx) < x_slope_threshold #, max_iterations=2, min_sigma=mx_regression_tol)
        self.valid_pairs.x_slope=np.all(self.valid_segs.x_slope, axis=1) 
    
        # 5: define selected pairs
        self.valid_pairs.all=np.logical_and(self.valid_pairs.data.ravel(), np.logical_and(self.valid_pairs.y_slope.ravel(), self.valid_pairs.x_slope.ravel()))
        #### this next item is done in section 5.1.5
        #5a: identify unselected cycle segs 
#        self.cycle_not_used_for_fit=D6.cycle[np.logical_not(self.valid_pairs.all),:]
#        #self.unselected_cycle_segs=D6.cycle[np.where(np.logical_not(self.valid_pairs.all)),:]
#        print('cycle not used for fit',self.cycle_not_used_for_fit,np.unique(self.cycle_not_used_for_fit))
        
#        self.unselected_cycle_segs=np.in1d(D6.cycle, unselected_cycles).reshape(D6.cycle.shape)  # (x,2)
#        print('self.unselected_cycle_segs',self.unselected_cycle_segs)
        return
        
    def select_y_center(self, D6, pair_data, params_11):  #5.1.3
        cycle=D6.cycle[self.valid_pairs.all,:]
        # find the middle of the range of the selected beams
        y0=(np.min(D6.y_atc[self.valid_pairs.all,:].ravel())+np.max(D6.y_atc[self.valid_pairs.all,:].ravel()))/2
        # 1: define a range of y centers, select the center with the best score
        y0_shifts=np.round(y0)+np.arange(-100,100, 2)
        score=np.zeros_like(y0_shifts)

        # 2: search for optimal shift val.ue
        for count, y0_shift in enumerate(y0_shifts):
            sel_segs=np.all(np.abs(D6.y_atc[self.valid_pairs.all,:]-y0_shift)<params_11.L_search_XT, axis=1)
            sel_cycs=np.unique(cycle[sel_segs,0])
            selected_seg_cycle_count=len(sel_cycs)
            
            other_passes=np.unique(cycle.ravel()[~np.in1d(cycle.ravel(),sel_cycs)])
            unsel_segs=np.logical_and(np.in1d(cycle.ravel(),other_passes),np.abs(D6.y_atc[self.valid_pairs.all,:].ravel()-y0_shift)<params_11.L_search_XT)
            unsel_cycs=np.unique(cycle.ravel()[unsel_segs])
            unselected_seg_cycle_count=len(unsel_cycs)
            
            # the score is equal to the number of cycles with at least one valid pair entirely in the window, 
            # plus 1/100 of the number cycles that contain no valid pairs but have at least one valid segment in the window
            score[count]=selected_seg_cycle_count + unselected_seg_cycle_count/100.
        # 3: identify the y0_shift value that corresponds to the best score, y_best, formally y_atc_ctr
        best = np.argwhere(score == np.amax(score))
        self.y_atc_ctr=np.median(y0_shifts[best])
        if self.DOPLOT is not None:        
            plt.figure(2);plt.clf()
            plt.plot(y0_shifts,score,'.');
            plt.plot(np.ones_like(np.arange(1,np.amax(score)+1))*self.y_atc_ctr,np.arange(1,np.amax(score)+1),'r')
            plt.title(' score vs y0_shifts(blu), y_best(red)')
        
        # 4: update valid pairs to inlucde y_atc within L_search_XT of y_atc_ctr (y_best)
        self.valid_pairs.ysearch=np.logical_and(self.valid_pairs.ysearch,np.abs(pair_data.y - self.y_atc_ctr)<params_11.L_search_XT)  
        if self.DOPLOT is not None:
            plt.figure(50)
            plt.plot(pair_data.y,'b.');plt.hold(True)
            plt.plot(pair_data.y[self.valid_pairs.data],'r.')
            plt.plot(D6.y_atc,'o');
            plt.plot(D6.y_atc[self.valid_pairs.all,:],'+');plt.grid(True)
        #plt.figure(51);plt.clf()
        #plt.plot(np.abs(pair_data.y - y_atc_ctr)<params_11.L_search_XT[self.valid_pairs.data])


        return 

    def find_reference_surface(self, D6, params_11):  #5.1.4
        # establish some output variables
        self.pass_h_shapecorr=np.full(12, np.nan)
        self.pass_h_shapecorr_sigma=np.full(12, np.nan)
        self.pass_lon=np.full(12, np.nan)
        self.pass_lat=np.full(12, np.nan)
        self.pass_x=np.full(12, np.nan)
        self.pass_y=np.full(12, np.nan)
        self.complex_surface_flag=0
        
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
        D_repeat=sparse.csc_matrix((data,(row,col)),shape=[len(cycle),len(self.ref_surf_passes)])    
        self.repeat_cols=np.arange(D_repeat.toarray().shape[1])
        
        # 2. determine polynomial degree, using unique x's and unique y's of segments in valid pairs
        x_atcU = np.unique(D6.x_atc[self.valid_pairs.all,:].ravel()) # np.unique orders the unique values
        y_atcU = np.unique(D6.y_atc[self.valid_pairs.all,:].ravel()) # np.unique orders the unique values
        poly_deg_x = np.minimum(params_11.poly_max_degree_AT,len(x_atcU)-1) 
        poly_deg_y = np.minimum(params_11.poly_max_degree_XT,len(y_atcU)-1) 

        # 3. perform an iterative fit for the across track polynomial
        # 3a. define degree_list_x and degree_list_y 
        self.x_degree_list, self.y_degree_list = np.meshgrid(np.arange(poly_deg_x+1), np.arange(poly_deg_y+1))
        # keep only degrees > 0 and degree_x+degree_y < max(max_x_degree, max_y_degree)
        sum_degrees=(self.x_degree_list + self.y_degree_list).ravel()
        keep=np.where(np.logical_and( sum_degrees <= np.maximum(poly_deg_x,poly_deg_y), sum_degrees > 0 ))
        self.x_degree_list = self.x_degree_list.ravel()[keep]
        self.y_degree_list = self.y_degree_list.ravel()[keep]
        sum_degree_list = sum_degrees[keep]
        # order by sum, x and then y
        degree_order=np.argsort(sum_degree_list + (self.y_degree_list / (self.y_degree_list.max()+1)))
        self.x_degree_list=self.x_degree_list[degree_order]
        self.y_degree_list=self.y_degree_list[degree_order]
        
        # 3b. define polynomial matrix
        print('x_ctr is',self.x_atc_ctr)
        print('y_ctr is',self.y_atc_ctr)
        x_atc=D6.x_atc[self.valid_pairs.all,:].ravel()
        y_atc=D6.y_atc[self.valid_pairs.all,:].ravel()
        S_fit_poly=np.zeros((len(x_atc),len(self.x_degree_list)),dtype=float)
        for jj in range(len(x_atc)):
            for ii in range(len(self.x_degree_list)):
                x_term=( (x_atc[jj]-self.x_atc_ctr)/params_11.xy_scale )**self.x_degree_list[ii]
                y_term=( (y_atc[jj]-self.y_atc_ctr)/params_11.xy_scale )**self.y_degree_list[ii]
                S_fit_poly[jj,ii]=x_term*y_term                
            
        # 3c. define slope-change matrix 
        # 3d. build the fitting matrix
        delta_time=D6.delta_time[self.valid_pairs.all,:].ravel()
        t_ctr=(np.max(delta_time)-np.min(delta_time))/2  # mid-point between start and end of mission

##### comment out when testing self.slope_change_rate
        if (np.max(delta_time)-np.min(delta_time))/params_11.t_scale > 1.5:
            x_term=np.array( [(x_atc-self.x_atc_ctr)/params_11.xy_scale * (delta_time-t_ctr)/params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/params_11.xy_scale * (delta_time-t_ctr)/params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_full=np.concatenate( (S_fit_poly,S_fit_slope_change,D_repeat.toarray()),axis=1 ) # G = [S St D]
            self.poly_cols=np.arange(S_fit_poly.shape[1])
            self.slope_change_cols=np.arange(S_fit_slope_change.shape[1])
        else:
            G_full=np.concatenate( (S_fit_poly,D_repeat.toarray()),axis=1 ) # G = [S D]
            self.poly_cols=np.arange(S_fit_poly.shape[1])
            self.slope_change_cols=np.array([])
            
        max_fit_iterations = 20  
        fit_columns=np.ones(G_full.shape[1],dtype=bool)

        # reduce these variables to the segments of the valid pairs only
        h_li_sigma=D6.h_li_sigma[self.valid_pairs.all,:].ravel() 
        h_li      =D6.h_li[self.valid_pairs.all,:].ravel()
            
        for kk in range(max_fit_iterations): 
            G=G_full[selected_segs,:]
            # 3e. Subset fitting matrix, to include columns that are not uniform
            for c in range(G.shape[1]-1,-1,-1):   # check lat col first, do in reverse order
                if np.all(G[:,c]==G[0,c]):
                    fit_columns[c]=False
                    G=np.delete(G,[c],1)
            # if three or more cycle columns are lost, use planar fit in x and y (end of section 3.3)
            if np.sum(np.logical_not(fit_columns[np.sum(self.poly_cols.shape,self.slope_change_cols.shape):])) > 2: 
                S_fit_poly=np.zeros((len(x_atc),2),dtype=float)
                for jj in range(len(x_atc)):
                    x_term=( (x_atc[jj]-self.x_atc_ctr)/params_11.xy_scale )
                    y_term=( (y_atc[jj]-self.y_atc_ctr)/params_11.xy_scale )
                    S_fit_poly[jj,0]=x_term
                    S_fit_poly[jj,1]=y_term
                self.poly_cols=np.arange(S_fit_poly.shape[1])
                self.slope_change_cols=np.array([])
                G_full=np.concatenate( (S_fit_poly,D_repeat.toarray()),axis=1 )

                fit_columns=np.ones(G_full.shape[1],dtype=bool)
                Cd, Cdi, G_g = gen_inv(self,G_full,h_li_sigma)
                m_ref=np.zeros(np.size(G_full,1))            
                m_ref=np.dot(G_g,h_li)  
                selected_pairs=np.ones( (np.sum(self.valid_pairs.all),),dtype=bool) 
                selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
                r_seg=h_li-np.dot(G_full,m_ref)
                r_fit=r_seg
                self.complex_surface_flag=1
                break
            # 3f. generate the data-covariance matrix
            # 3g. equation 7
            Cd, Cdi, G_g = gen_inv(self,G,h_li_sigma[selected_segs])
                        
            # inititalize the reference model 
            m_ref=np.zeros(np.size(G_full,1))            
            z=h_li[selected_segs]            
            m_ref[fit_columns]=np.dot(G_g,z)  

            # 3h. Calculate model residuals for all segments
            r_seg=h_li-np.dot(G_full,m_ref)            
            r_fit=r_seg[selected_segs] 
            
            # 3i. Calculate the fitting tolerance, 
            r_tol = 3*RDE(r_fit/h_li_sigma[selected_segs])
            # reduce chi-squared value
            surf_fit_misfit_chi2 = np.dot(np.dot(np.transpose(r_fit),Cdi.toarray()),r_fit)

            # calculate P value
            n=np.sum(fit_columns)
            m=np.sum(selected_segs)
            P = 1 - stats.chi2.cdf(surf_fit_misfit_chi2,m-n)

            # 3j. 
            if P<0.025:
                selected_segs = np.abs(r_seg/h_li_sigma) < r_tol # boolean
            selected_segs=np.reshape(selected_segs,(len(selected_pairs),2))
            selected_pairs=np.logical_and(selected_segs[:,0], selected_segs[:,1])
            selected_segs=np.column_stack((selected_pairs,selected_pairs)).ravel()
        
        segment_id=D6.segment_id[self.valid_pairs.all,:].ravel()[selected_segs]
        x_atc=D6.x_atc[self.valid_pairs.all,:].ravel()[selected_segs]
        y_atc=D6.y_atc[self.valid_pairs.all,:].ravel()[selected_segs]
        lon=D6.longitude[self.valid_pairs.all,:].ravel()[selected_segs]
        lat=D6.latitude[self.valid_pairs.all,:].ravel()[selected_segs]
        time=D6.delta_time[self.valid_pairs.all,:].ravel()[selected_segs]
        h_li_sigma=D6.h_li_sigma[self.valid_pairs.all,:].ravel()[selected_segs]
        h_li      =D6.h_li[self.valid_pairs.all,:].ravel()[selected_segs]
        
        cycle=D6.cycle[self.valid_pairs.all,:].ravel()[selected_segs]
        self.ref_surf_passes=self.ref_surf_passes[fit_columns[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols]]
        
        if self.DOPLOT is not None:
            fig=plt.figure(31); plt.clf(); ax=fig.add_subplot(111, projection='3d')        
            p=ax.scatter(x_atc, y_atc, h_li, c=time); 
            fig.colorbar(p)
        
        # separate m_ref
        self.ref_surf_poly=m_ref[self.poly_cols]
        if self.slope_change_cols.shape[0]>0:
            self.ref_surf_slope_change_rate=m_ref[self.poly_cols.shape[0]+self.slope_change_cols]
        else:
            self.ref_surf_slope_change_rate=np.zeros((2,))
        self.z_cycle=m_ref[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols] # the 'intercept'
        self.pass_h_shapecorr[self.ref_surf_passes.astype(int)-1]=self.z_cycle[fit_columns[np.sum([self.poly_cols.shape,self.slope_change_cols.shape]):]] #np.sum([self.poly_cols.shape,self.slope_change_cols.shape,self.repeat_cols])]] )

        for cc in self.ref_surf_passes:
            self.pass_lon[cc.astype(int)-1]=np.mean(lon[(cycle==cc)])
            self.pass_lat[cc.astype(int)-1]=np.mean(lat[(cycle==cc)])
            self.pass_x[cc.astype(int)-1]=np.mean(x_atc[(cycle==cc)])
            self.pass_y[cc.astype(int)-1]=np.mean(y_atc[(cycle==cc)])
        
        # 3k. propagate the errors   
        Cdp=sparse.diags(np.maximum(h_li_sigma**2,(RDE(r_fit))**2))  # C1 in text  
        self.Cm = np.dot(np.dot(G_g,Cdp.toarray()),np.transpose(G_g))
        self.sigma_m=np.full(G_full.shape[1],np.nan)
        self.sigma_m[fit_columns]=np.sqrt(self.Cm.diagonal())
        plt.figure(3);plt.clf()
        plt.plot(self.sigma_m[:np.sum(self.poly_cols.shape,self.slope_change_cols.shape)],'ro-')
        plt.hold(True)
        plt.plot(m_ref[:np.sum(self.poly_cols.shape,self.slope_change_cols.shape)],'go-')
        plt.xticks(np.arange(9),(self.x_degree_list+self.y_degree_list).astype('S3'))
        plt.xlabel('sum of x_degree, y degree')
        plt.title('Surface Shape Polynomial (g), Sigma m (r)')
        self.z_cycle_sigma=self.sigma_m[self.poly_cols.shape[0]+self.slope_change_cols.shape[0]+self.repeat_cols] # the 'intercept'
        self.pass_h_shapecorr_sigma[self.ref_surf_passes.astype(int)-1]=self.z_cycle_sigma[fit_columns[np.sum([self.poly_cols.shape,self.slope_change_cols.shape]):]] #np.sum([self.poly_cols.shape,self.slope_change_cols.shape,self.repeat_cols])]] )
        
        return 
    
    def corr_heights_other_cycles(self, D6, params_11):
        # find cycles not in ref_surface_passes 
        other_passes=np.unique(D6.cycle.ravel()[~np.in1d(D6.cycle.ravel(),self.ref_surf_passes)])
        # 1. find cycles not in ref_surface_passes, but have valid_segs.data and valid_segs.x_slope  
        non_ref_segments=np.logical_and(np.in1d(D6.cycle.ravel(),other_passes),np.logical_and(self.valid_segs.data.ravel(),self.valid_segs.x_slope.ravel()))
        cycles=D6.cycle.ravel()[non_ref_segments]
    
        # 2. build design matrix, G_other, for non selected segments (poly and dt parts only)
        x_atc=D6.x_atc.ravel()[non_ref_segments]
        y_atc=D6.y_atc.ravel()[non_ref_segments]
        if self.complex_surface_flag==0:
            S_fit_poly=np.zeros((len(x_atc),len(self.x_degree_list)),dtype=float)
            for jj in range(len(x_atc)):
                for ii in range(len(self.x_degree_list)):
                    x_term=( (x_atc[jj]-self.x_atc_ctr)/params_11.xy_scale )**self.x_degree_list[ii]
                    y_term=( (y_atc[jj]-self.y_atc_ctr)/params_11.xy_scale )**self.y_degree_list[ii]
                    S_fit_poly[jj,ii]=x_term*y_term            
        else:
            S_fit_poly=np.zeros((len(x_atc),2),dtype=float)
            for jj in range(len(x_atc)):
                x_term=( (x_atc[jj]-self.x_atc_ctr)/params_11.xy_scale )
                y_term=( (y_atc[jj]-self.y_atc_ctr)/params_11.xy_scale )
                S_fit_poly[jj,0]=x_term
                S_fit_poly[jj,1]=y_term
            
        delta_time=D6.delta_time.ravel()[non_ref_segments]
        t_ctr=(np.max(delta_time)-np.min(delta_time))/2  # mid-point between start and end of mission
#### needs to be uncommented with real data
        if (np.max(delta_time)-np.min(delta_time))/params_11.t_scale > 1.5:
            x_term=np.array( [(x_atc-self.x_atc_ctr)/params_11.xy_scale * (delta_time-t_ctr)/params_11.t_scale] )
            y_term=np.array( [(y_atc-self.y_atc_ctr)/params_11.xy_scale * (delta_time-t_ctr)/params_11.t_scale] )
            S_fit_slope_change=np.concatenate((x_term.T,y_term.T),axis=1)
            G_other=np.concatenate( (S_fit_poly,S_fit_slope_change),axis=1 ) # G [S St]
        else:
            G_other=S_fit_poly  #  G=[S]
         
        # with heights and errors from non_ref_segments 
        h_li      =D6.h_li.ravel()[non_ref_segments]
        h_li_sigma=D6.h_li_sigma.ravel()[non_ref_segments]
        cycle=D6.cycle.ravel()[non_ref_segments]
        lon=D6.longitude.ravel()[non_ref_segments]
        lat=D6.latitude.ravel()[non_ref_segments]
        self.non_ref_surf_passes=np.unique(cycle)
        
        # calculate corrected heights, z_kc, with non selected segs design matrix and surface shape polynomial from selected segments
        if self.slope_change_cols.shape[0]>0:
            z_kc=h_li - np.dot(G_other,np.concatenate((self.ref_surf_poly,self.ref_surf_slope_change_rate))) # equation 10
        else:
            z_kc=h_li - np.dot(G_other,self.ref_surf_poly) 
          
        plt.figure(107);plt.clf()
        plt.plot(h_li,'b.-');plt.hold(True)
        plt.plot(z_kc,'ro-')
        plt.title('Other cycles: hli (b), Zkc-Corrected Heights (r)');plt.grid()
        
        # use errors from surface shape polynomial and non-selected segs design matrix to get non selected segs height corrs errors
        if self.slope_change_cols.shape[0]>0:
            Cms=self.Cm[:,np.concatenate( (self.poly_cols,self.slope_change_cols) )][np.concatenate( (self.poly_cols,self.slope_change_cols) ),:] 
        else:
            Cms=self.Cm[:,self.poly_cols][self.poly_cols,:] # can't index 2 dimensions at once. IF NO SLOPE_CHANGE_COLS!!
        z_kc_sigma = np.sqrt( np.diag( np.dot(np.dot(G_other,Cms),np.transpose(G_other)) ) + h_li_sigma**2 ) # equation 11
        plt.figure(108);plt.clf()
        plt.plot(h_li_sigma,'b.-');plt.hold(True)
        plt.plot(z_kc_sigma,'ro-')
        plt.title('Other cycles: hli sigma (b), Zkc sigma(r)');plt.grid()
        
        for cc in self.non_ref_surf_passes:
            self.pass_h_shapecorr[np.int(cc)-1]=z_kc[cycle==cc][np.argmin(z_kc_sigma[cycle==cc])]
            self.pass_h_shapecorr_sigma[np.int(cc)-1]=np.amin(z_kc_sigma[cycle==cc])
            self.pass_lon[np.int(cc)-1]=lon[cycle==cc][np.argmin(z_kc_sigma[cycle==cc])]
            self.pass_lat[np.int(cc)-1]=lat[cycle==cc][np.argmin(z_kc_sigma[cycle==cc])]
            self.pass_x[np.int(cc)-1]  =x_atc[cycle==cc][np.argmin(z_kc_sigma[cycle==cc])]
            self.pass_y[np.int(cc)-1]  =y_atc[cycle==cc][np.argmin(z_kc_sigma[cycle==cc])]

        plt.figure(200);plt.clf()
        plt.plot(np.arange(12)+1,self.pass_h_shapecorr,'bo-');plt.hold(True)
        plt.plot(np.arange(12)[self.non_ref_surf_passes.astype(int)-1]+1,self.pass_h_shapecorr[self.non_ref_surf_passes.astype(int)-1],'ro')
        plt.xlabel('Cycle Number');plt.xlim((0,13))
        plt.ylabel('Corrected Height with lowest Error / Cycle')
        plt.title('Pass H ShapeCorr: selected (b), other (r)');plt.grid()
        plt.figure(201);plt.clf()
        plt.plot(np.arange(12)+1,self.pass_h_shapecorr_sigma,'bo-');plt.hold(True)
        plt.plot(np.arange(12)[self.non_ref_surf_passes.astype(int)-1]+1,self.pass_h_shapecorr_sigma[self.non_ref_surf_passes.astype(int)-1],'ro')
        plt.xlabel('Cycle Number');plt.xlim((0,13))
        plt.ylabel('Lowest Error of Segments in each Cycle')
        plt.title('Pass H ShapeCorr Sigma: selected (b), other (r)');plt.grid()
        
def gen_inv(self,G,sigma):
            # 3f. Generate data-covariance matrix
    Cd=sparse.diags(sigma**2)
    Cdi=sps_linalg.inv(Cd)
    G_sq=np.dot(np.dot(np.transpose(G),Cdi.toarray()),G)
            
    G_sqi=linalg.inv(G_sq)
            # calculate the generalized inverse of G
    G_g=np.dot( np.dot(G_sqi,np.transpose(G)),Cdi.toarray() )  
            
    return Cd, Cdi, G_g
        
         
class ATL11_defaults:
    def __init__(self):
        # provide option to read keyword=val pairs from the input file
        self.L_search_AT=125 # meters, along track (in x), filters along track
        self.L_search_XT=110 # meters, cross track (in y), filters across track
        self.min_slope_tol=0.02 # in degrees?
        self.min_h_tol=0.1  # units? of height offset
        self.seg_sigma_threshold_min=0.05
        #self.y_search=110 # meters
        self.beam_spacing=90 # meters
        self.seg_atc_spacing=20 # meters, segments are 40m long, overlap is 50%
        self.poly_max_degree_AT=3
        self.poly_max_degree_XT=3
        self.xy_scale=100.         # meters
        self.t_scale=86400*365.25  # assuming 365.25 days in one year. t_scale in seconds.
        
        
        
        