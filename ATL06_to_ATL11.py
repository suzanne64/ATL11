import numpy as np

from ATL06_data import ATL06_data

from ATL11 import ATL11_data, ATL11_point, ATL11_defaults
import matplotlib.pyplot as plt 
import time

from mpl_toolkits.mplot3d import Axes3D

#from poly_ref_surf import poly_ref_surf

def fit_ATL11(ATL06_files, beam_pair=1, seg_x_centers=None, output_file=None, num_centers=None, DOPLOT=None, DEBUG=None):
    params_11=ATL11_defaults()

    # read in the ATL06 data from all the repeats
    D6=ATL06_data(filename=ATL06_files, beam_pair=beam_pair, NICK=True) # two cols (two segs)
    # reoder data rows from D6 by cycle
    this_index=np.argsort(D6.cycle[:,0],axis=0)
    for field in D6.list_of_fields:
        tmp=getattr(D6,field)
        setattr(D6,field,tmp[this_index, :])
        
    P11_list=list()
    if seg_x_centers is None:
        # NO: select every nth center        
        seg_x_centers=np.arange(np.min(np.c_[D6.x_atc]), np.max(np.c_[D6.x_atc]), params_11.seg_atc_spacing) 
    if num_centers is not None:
        seg_x_centers=seg_x_centers[0:int(num_centers)]
    for seg_x_center in seg_x_centers:
        # section 5.1.1 ?
        D6_sub=D6.subset(np.any(np.abs(D6.x_atc-seg_x_center) < params_11.L_search_AT, axis=1), by_row=True) # len 144 = 12 xlocs, by 12 cycles where ylocs are diff for each cycle, xlocs the same for all cycles.
        if DEBUG:
            print('seg_x_center',seg_x_center)
            print('D6 sub shape',D6_sub.x_atc.shape)
        #2a. define representative x and y values for the pairs
        pair_data=D6_sub.get_pairs()   # this might go, similar to D6_sub

        P11=ATL11_point(N_pairs=len(pair_data.x), x_atc_ctr=seg_x_center, y_atc_ctr=None, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),N_reps=len(ATL06_files),N_coeffs=9 )
        
        P11.DOPLOT=DOPLOT
       # step 2: select pairs, based on reasonable slopes
        P11.select_ATL06_pairs(D6_sub, pair_data, params_11)
        if 'no_valid_pairs' in P11.status and P11.status['no_valid_pairs']==1:
            print('you have no valid pairs',seg_x_center)
            continue
        P11.select_y_center(D6_sub, pair_data, params_11)
                
        P11.D.ref_pt_lat,P11.D.ref_pt_lon = regress_to(D6_sub,['latitude','longitude'], ['x_atc','y_atc'],[seg_x_center,P11.y_atc_ctr])
        
        P11.find_reference_surface(D6_sub, params_11)
        
        P11.corr_heights_other_cycles(D6_sub, params_11)
        
        P11_list.append(P11)
        
        
    return P11_list
  
def regress_to(D, out_field_names, in_field_names, in_field_pt, DEBUG=None):
    D_in =np.transpose( np.array((getattr(D, in_field_names[0]).ravel(), getattr(D, in_field_names[1]).ravel())) )
    D_out=np.transpose( np.array((getattr(D,out_field_names[0]).ravel(), getattr(D,out_field_names[1]).ravel())) )
    good_rows=np.all(~np.isnan( np.concatenate( (D_in,D_out),axis=1)), axis=1)

    G=np.ones((np.sum(good_rows),len(in_field_names)+1) )
    for k in range(len(in_field_names)):
        G[:,k+1]= np.subtract(getattr(D,in_field_names[k]).ravel()[good_rows] , in_field_pt[k])

    out_pt0 = np.linalg.lstsq(G,getattr(D,out_field_names[0]).ravel()[good_rows])[0]
    out_pt1 = np.linalg.lstsq(G,getattr(D,out_field_names[1]).ravel()[good_rows])[0]
    if DEBUG is not None:    
        print('lat_ctr,lon_xtr',out_pt0[0],out_pt1[0])

    return out_pt0[0],out_pt1[0]
    

