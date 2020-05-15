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
        #self.L_search_AT=100 # meters, along track (in x), filters along track
        self.L_search_AT=60
        self.L_search_XT=65 # meters, cross track (in y), filters across track
        self.seg_sigma_threshold_min=0.05
        self.beam_spacing=90 # meters
        self.seg_atc_spacing=100 # meters
        self.N_search=self.L_search_AT/20 # segments to include +/- from seg_atc_ctr in each ref_pt analysis
        self.poly_max_degree_AT=3
        self.poly_max_degree_XT=2
        self.xy_scale=100.         # meters
        self.t_scale=86400*365.25  # seconds/year.
        self.max_fit_iterations = 20  # maximum iterations when computing the reference surface models
        self.equatorial_radius=6378137 # meters, on WGS84 spheroid
        self.polar_radius=6356752.3 # derived, https://www.eoas.ubc.ca/~mjelline/Planetary%20class/14gravity1_2.pdf
        self.ATL06_field_dict=default_ATL06_fields()
        self.seg_number_skip=self.N_search

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
        self.poly_exponent={'x':degree_list_x, 'y':degree_list_y}
        self.N_coeffs=len(self.poly_exponent_list)
        self.hemisphere=None
        self.ATL06_xover_field_list=['delta_time','h_li','h_li_sigma','latitude',\
                                     'longitude','atl06_quality_summary','segment_id',\
                                     'x_atc', 'dh_fit_dx', 'rgt','cycle_number',\
                                     'BP', 'LR', 'spot','sigma_geo_xt','sigma_geo_at', \
                                     'sigma_geo_h']
        

def default_ATL06_fields():
    # NOTE: when release 3 comes out, change sigma_geo_r to a parameter in the ground_track group
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude',
                      'atl06_quality_summary','segment_id','sigma_geo_h'],
                    'ground_track':['x_atc', 'y_atc','seg_azimuth','sigma_geo_at',
                                    'sigma_geo_xt'],
                    'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','h_mean', 
                                      'dh_fit_dy','h_rms_misfit','h_robust_sprd',
                                      'n_fit_photons', 'signal_selection_source',
                                      'snr_significance','w_surface_window_final'],
                    'geophysical':['bsnow_conf','bsnow_h','cloud_flg_asr',
                                   'cloud_flg_atm','r_eff','tide_ocean','dac'],
                    'orbit_info':['rgt','cycle_number'],
                    'dem':['dem_h'],
                    'derived':['valid','BP', 'LR', 'spot', 'n_pixels',
                               'min_along_track_dh', 'sigma_geo_r']}
    return field_dict

