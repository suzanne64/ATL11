# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from ATL11.RDE import RDE


class defaults:
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
        self.ATL06_field_dict=default_ATL06_fields()

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

def default_ATL06_fields():
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'],
                    'ground_track':['x_atc', 'y_atc','seg_azimuth','sigma_geo_at','sigma_geo_xt'],
                    'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','h_mean', 'dh_fit_dy','h_rms_misfit','h_robust_sprd','n_fit_photons', 'signal_selection_source','snr_significance','w_surface_window_final'],
                    'geophysical':['bsnow_conf','bsnow_h','cloud_flg_asr','cloud_flg_atm','r_eff','tide_ocean'],
                    'orbit_info':['rgt','cycle_number'],
                    'derived':['valid','BP', 'LR']}
    return field_dict

class poly_ref_surf(object):
    def __init__(self, degree_xy=None, exp_xy=None, xy0=[0,0], skip_constant=False, xy_scale=1.0):  
        self.x0=xy0[0]
        self.y0=xy0[1]
        if degree_xy is not None:
            poly_exp_x, poly_exp_y=np.meshgrid(np.arange(0, degree_xy[0]+1), np.arange(0, degree_xy[1]+1))
            temp=np.asarray(list(set(zip(poly_exp_x.ravel(), poly_exp_y.ravel()))))  # makes one array out of two
            # Remove exponents with exp_x+exp_y larger that max(exp_x, exp_y)
            temp=temp[np.sum(temp, axis=1)<=np.array(degree_xy).max(),:];    
            # if the skip_constant option is chosen, eliminate the constant term        
            if skip_constant:
                temp=temp[np.all(temp>0, axis=1)]
            # sort the exponents first by x, then by y    
            temp=temp[(temp[:,0]+temp[:,1]/(temp.shape[0]+1.)).argsort()]  # orders the array: 0,0  0,1  1,0
            self.exp_x=temp[:,0]  # [0,0,1]
            self.exp_y=temp[:,1]  # [0,1,0]
        if exp_xy is not None:
            self.exp_x=np.array(exp_xy[0]).ravel()
            self.exp_y=np.array(exp_xy[1]).ravel()
        # establish arrays for fitting
        self.poly_vals=np.NaN+np.zeros(self.exp_x.shape)
        self.model_cov_matrix=None
        self.xy_scale=xy_scale
        self.skip_constant=skip_constant
    def fit_matrix(self, x, y):
        G=np.zeros([x.size, self.exp_x.size])  # size is the ravel of shape, here G is len(x) * 3
        for col, ee in enumerate(zip(self.exp_x, self.exp_y)):
            G[:,col]=((x.ravel()-self.x0)/self.xy_scale)**ee[0] * ((y.ravel()-self.y0)/self.xy_scale)**ee[1]
        return G
    def z(self, x0, y0):
        # evaluate the poltnomial at [x0, y0]
        G=self.fit_matrix(x0, y0)
        z=np.dot(G, self.poly_vals)
        z.shape=x0.shape
        return z
    def fit(self, xd, yd, zd, sigma_d=None, max_iterations=1, min_sigma=0):
             
        # asign poly_vals and cov_matrix with a linear fit to zd at points xd, yd
        # build the design matrix:      
        G=self.fit_matrix(xd, yd)
        #initialize outputs
        m=np.zeros(G.shape[1])+np.NaN
        residual=np.zeros_like(xd)+np.NaN
        rows=np.ones_like(xd, dtype=bool)
        # build a sparse covariance matrix
        if sigma_d is None:
            sigma_d=np.ones_like(xd.ravel())
        chi2r=len(zd)*2
        mask=np.ones_like(zd.ravel(), dtype=bool)
        for k_it in np.arange(max_iterations):
            rows=mask
            if rows.sum()==0:
                chi2r=np.NaN
                break
            sigma_inv=sparse.diags(1/sigma_d.ravel()[rows])
            Gsub=G[rows,:]
            cols=(np.amax(Gsub,axis=0)-np.amin(Gsub,axis=0))>0
            if self.skip_constant is False:
                cols[0]=True
            m=np.zeros([Gsub.shape[1],1])
            # compute the LS coefficients
            msub, rr, rank, sing=linalg.lstsq(sigma_inv.dot(Gsub[:,cols]), sigma_inv.dot(zd.ravel()[rows]))
            msub.shape=(len(msub), 1)
            m[np.where(cols)]=msub  # only takes first three coefficients?
            residual=zd.ravel()-G.dot(m).ravel()
            rs=residual/sigma_d.ravel()
            chi2r_last=chi2r
            deg_of_freedom=(rows.sum()-cols.sum())
            if deg_of_freedom  > 0:
                chi2r=sum(rs**2)/(deg_of_freedom)     
            else:
                # the inversion is even-determined or worse, no further improvement is expected
                break
            #print('chi2r is ',chi2r)
            if np.abs(chi2r_last-chi2r)<0.01 or chi2r<1:
                break
            sigma=RDE(rs[rows])
            if not np.isfinite(sigma):
                sigma=0
            print('sigma from RDE ',sigma)
            threshold=3.*np.max([sigma, min_sigma])
            mask=np.abs(rs)<threshold
            print("\tsigma=%3.2f, f=%d/%d" % (sigma, np.sum(mask), len(mask)))
            # In the future, compute the LS coefficients using PySPQR (get from github.com/yig/PySPQR)
        self.poly_vals=m
        
        return m, residual, chi2r, rows
