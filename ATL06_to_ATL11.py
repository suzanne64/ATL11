import numpy as np
from ATL06_data import ATL06_data
from ATL11_point import ATL11_point, 
from ATL11_misc import ATL11_defaults


from mpl_toolkits.mplot3d import Axes3D

#from poly_ref_surf import poly_ref_surf

def fit_ATL11(ATL06_files, beam_pair=1, ref_pt_numbers=None, output_file=None, num_ref_pts=None, DOPLOT=None, DEBUG=None, mission_time_bds=None):
    params_11=ATL11_defaults()
    seg_number_skip=int(params_11.seg_atc_spacing/20);
    if mission_time_bds is None:
        mission_time_bds=np.array([0, 365.25*3.*24.*3600.])

    
    # read in the ATL06 data from all the repeats
    D6=ATL06_data(filename=ATL06_files, beam_pair=beam_pair, NICK=True)
    
    # reoder data rows from D6 by cycle
    this_index=np.argsort(D6.cycle[:,0],axis=0)
    for field in D6.list_of_fields:
        tmp=getattr(D6,field)
        setattr(D6,field,tmp[this_index, :])
    
    P11_list=list()
    if ref_pt_numbers is None:
        uId, iId=np.unique(D6.segment_id.ravel(), return_index=True)
        ctrSegs=np.mod(uId, seg_number_skip)==0
        ref_pt_numbers=uId[ctrSegs]
        ref_pt_x=D6.x_atc.ravel()[iId[ctrSegs]]
    else:
        ref_pt_x=ref_pt_numbers*20
        
    if num_ref_pts is not None:
        ref_pt_numbers=ref_pt_numbers[0:int(num_ref_pts)]
        ref_pt_x=ref_pt_x[0:int(num_ref_pts)]

    for count, ref_pt_number in enumerate(ref_pt_numbers):
        x_atc_ctr=ref_pt_x[count]
        # section 5.1.1 
        D6_sub=D6.subset(np.any(np.abs(D6.segment_id-ref_pt_number) <= params_11.N_search, axis=1), by_row=True)
        if D6_sub.h_li.shape[0]<=1:
            continue
        
        #2a. define representative x and y values for the pairs
        pair_data=D6_sub.get_pairs(datasets=['x_atc','y_atc','delta_time','dh_fit_dx','dh_fit_dy','segment_id','cycle','h_li'])   # this might go, similar to D6_sub

        P11=ATL11_point(N_pairs=len(pair_data.x), ref_pt_number=ref_pt_number, x_atc_ctr=x_atc_ctr, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),N_reps=len(ATL06_files),  mission_time_bds=mission_time_bds )
         
        P11.DOPLOT=DOPLOT
       # step 2: select pairs, based on reasonable slopes
        P11.select_ATL06_pairs(D6_sub, pair_data)
        if 'no_valid_pairs' in P11.status and P11.status['no_valid_pairs']==1:
            #print('you have no valid pairs',seg_x_center)
            continue
        P11.select_y_center(D6_sub, pair_data)
        
        P11.corrected_h.ref_pt_lat,P11.corrected_h.ref_pt_lon = regress_to(D6_sub,['latitude','longitude'], ['x_atc','y_atc'],[x_atc_ctr,P11.y_atc_ctr])     

        P11.find_reference_surface(D6_sub)
        if 'inversion failed' in P11.status:
            continue

        P11.corr_heights_other_cycles(D6_sub)

 
        P11_list.append(P11)
        if np.mod(count, 1)==100:
            print("completed %d segments, ref_pt_number= %d" %(count, ref_pt_number))    
        
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
    

