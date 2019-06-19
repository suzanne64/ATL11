
import os
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
os.environ['OPENBLAS_NUM_THREADS']="1"

import numpy as np
from PointDatabase.ATL06_data import ATL06_data
from PointDatabase.point_data import point_data
from PointDatabase.geo_index import geo_index
import ATL11
import glob
import sys
import matplotlib.pyplot as plt

def get_xover_data(x0, y0, rgt, GI_file, xover_cache, delta_bins, index_bin_size, params_11):
    x0_ctrs=np.unique(np.round((x0+delta_bins['x'])/index_bin_size)*index_bin_size+1j*np.round((y0+delta_bins['y'])/index_bin_size)*index_bin_size)
    D_xover=[]
     
    for x0_ctr in x0_ctrs:
        this_key=(np.real(x0_ctr), np.imag(x0_ctr)) 
        # check if we have already read in the data for this bin
        if this_key not in xover_cache:
            # if we haven't already read in the data, read it in
            xover_cache[this_key]={'D':point_data(field_dict=params_11.ATL06_field_dict).from_list(geo_index().from_file(GI_file).query_xy(this_key, fields=params_11.ATL06_field_dict))}
            xover_cache[this_key]['D'].index(~np.in1d(xover_cache[this_key]['D'].rgt, [rgt]))
            if xover_cache[this_key]['D'].size==0:
                continue
            xover_cache[this_key]['D'].get_xy(EPSG=params_11.EPSG)
            xover_cache[this_key]['index']=geo_index(delta=[100, 100], data=xover_cache[this_key]['D'])            
        if xover_cache[this_key]['D'] is not None:  
            try:
                Q=xover_cache[this_key]['index'].query_xy([x0, y0], pad=1, get_data=False)
            except KeyError:
                Q=None
            if Q is None:
                continue
            # if we have read in any data for the current bin, subset it to the bins around the reference point
            for key in Q:
                for i0, i1 in zip(Q[key]['offset_start'], Q[key]['offset_end']):
                    D_xover.append(xover_cache[this_key]['D'].subset(np.arange(i0, i1+1, dtype=int)))
    if len(D_xover) > 0:
        D_xover=point_data().from_list(D_xover)
    return D_xover

def fit_ATL11(ATL06_files, beam_pair=1, N_cycles=2, ref_pt_numbers=None, output_file=None, num_ref_pts=None, first_ref_pt=None, last_ref_pt=None, DOPLOT=None, DEBUG=None, mission_time_bds=None, lonlat_bounds=None, verbose=False):
    params_11=ATL11.defaults()
    seg_number_skip=int(params_11.seg_atc_spacing/20);
    if mission_time_bds is None:
        mission_time_bds=np.array([286.*24*3600, 398.*24*3600])

    # read in the ATL06 data from all the repeats  
    D6_list=[] 
    for filename in ATL06_files: 
        try:
            D6_list.append(ATL06_data(field_dict=params_11.ATL06_field_dict, beam_pair=beam_pair).from_file(filename))
        except KeyError:
            pass
    if len(D6_list)==0:
        return None
    D6=ATL06_data(beam_pair=beam_pair).from_list(D6_list)

    if lonlat_bounds is not None:
        keep = (D6.longitude >= lonlat_bounds[0]) 
        keep &= (D6.latitude >= lonlat_bounds[1])
        keep &= (D6.longitude <= lonlat_bounds[2])
        keep &= (D6.latitude <= lonlat_bounds[3])
        keep = np.any(keep, axis=1)
        if not np.any(keep):
            return None
        D6.index(keep)

    # reorder data rows from D6 by cycle
    D6.index(np.argsort(D6.cycle_number[:,0],axis=0))
    if np.max(D6.latitude) < 0:
        D6.get_xy(None, EPSG=3031)
        params_11.EPSG=3031
        GI_file='/Volumes/ice2/ben/scf/AA_06/209/index/GeoIndex.h5'
        index_bin_size=1.e4
    else:
        D6.get_xy(None, EPSG=3413)
        
    # get list of reference points   
    if ref_pt_numbers is None:
        uId, iId=np.unique(D6.segment_id.ravel(), return_index=True)
        ctrSegs=np.mod(uId, seg_number_skip)==0
        ref_pt_numbers=uId[ctrSegs]
        ref_pt_x=D6.x_atc.ravel()[iId[ctrSegs]]
    else:
        ref_pt_x=ref_pt_numbers*20

    if first_ref_pt is not None:
        these=ref_pt_numbers>=first_ref_pt
        ref_pt_numbers=ref_pt_numbers[these]
        ref_pt_x=ref_pt_x[these]
        
    if last_ref_pt is not None:
        these=ref_pt_numbers<=last_ref_pt
        ref_pt_numbers=ref_pt_numbers[these]
        ref_pt_x=ref_pt_x[these]

    if num_ref_pts is not None:
        ref_pt_numbers=ref_pt_numbers[0:int(num_ref_pts)]
        ref_pt_x=ref_pt_x[0:int(num_ref_pts)]

    # initialize the xover data cache and the delta_x and delta_y values used to search across the edges of bins
    D_xover_cache={}
    delta_bins={}
    delta_bins['x'], delta_bins['y']=np.meshgrid(np.array([-1, 0, 1])*params_11.L_search_XT, np.array([-1, 0, 1])*params_11.L_search_XT)
    
    last_count=0
    # loop over reference points
    P11_list=list()
    for count, ref_pt_number in enumerate(ref_pt_numbers):
        
        x_atc_ctr=ref_pt_x[count]
        # section 5.1.1
        D6_sub=D6.subset(np.any(np.abs(D6.segment_id-ref_pt_number) <= params_11.N_search, axis=1), by_row=True)
        if D6_sub.h_li.shape[0]<=1:
            if verbose:
                print("not enough data at ref pt=%d" % ref_pt_number)
            continue

        #2a. define representative x and y values for the pairs
        pair_data=D6_sub.get_pairs(datasets=['x_atc','y_atc','delta_time','dh_fit_dx','dh_fit_dy','segment_id','cycle_number','h_li'])   # this might go, similar to D6_sub
        if ~np.any(np.isfinite(pair_data.y)):
            continue
        P11=ATL11.point(N_pairs=len(pair_data.x), rgt=D6_sub.rgt[0, 0], ref_pt_number=ref_pt_number, pair_num=D6_sub.BP[0, 0],  x_atc_ctr=x_atc_ctr, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),N_cycles=N_cycles,  mission_time_bds=mission_time_bds )

        P11.DOPLOT=DOPLOT
        # step 2: select pairs, based on reasonable slopes
        P11.select_ATL06_pairs(D6_sub, pair_data)
        if P11.ref_surf.surf_fit_quality_summary > 0:
            P11_list.append(P11)
            if verbose:
                print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
            continue

        P11.select_y_center(D6_sub, pair_data)
        if P11.ref_surf.surf_fit_quality_summary > 0:
            P11_list.append(P11)
            if verbose:
                print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
            continue

        P11.corrected_h.ref_pt_lat,P11.corrected_h.ref_pt_lon = regress_to(D6_sub,['latitude','longitude'], ['x_atc','y_atc'], [x_atc_ctr,P11.y_atc_ctr])

        P11.find_reference_surface(D6_sub)
        if 'inversion failed' in P11.status:
            P11_list.append(P11)
            if verbose:
                print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
            continue

        P11.corr_heights_other_cycles(D6_sub)

        # find the center of the bin in polar stereographic coordinates
        x0, y0=regress_to(D6_sub, ['x','y'], ['x_atc', 'y_atc'], [x_atc_ctr,P11.y_atc_ctr])
        
        # get the data for the crossover point
        D_xover=get_xover_data(x0, y0, P11.rgt, GI_file, D_xover_cache, delta_bins, index_bin_size, params_11)
        # if we have read any data for the current bin, run the crossover calculation
        PLOTME=False#isinstance(D_xover, point_data);
        if PLOTME:
            plt.figure()
            for key in D_xover_cache.keys():
                plt.plot(D_xover_cache[key]['D'].x, D_xover_cache[key]['D'].y,'k.')
            
            plt.plot(D_xover.x, D_xover.y,'m.')
            plt.plot(x0, y0,'g*')

        P11.corr_xover_heights(D_xover)
        P11_list.append(P11)
        if count-last_count>500:
            print("completed %d segments, ref_pt_number= %d" %(count, ref_pt_number))
            last_count=count

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

def main(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg
   
    # command-line interface: run ATL06_to_ATL11 on a list of ATL06 files
    import argparse
    parser=argparse.ArgumentParser(description='generate an ATL11 file from a collection of ATL06 files.')
    parser.add_argument('rgt', type=int)
    parser.add_argument('subproduct', type=int)
    parser.add_argument('--directory','-d', default=os.getcwd())
    parser.add_argument('--verbose','-v', action='store_true')
    parser.add_argument('--pair','-p', type=int, default=None)
    parser.add_argument('--File', '-F', type=str, default=None)
    parser.add_argument('--out_file','-o', default=None, required=True)
    parser.add_argument('--first_point','-f', type=int, default=None)
    parser.add_argument('--last_point','-l', type=int, default=None)
    parser.add_argument('--cycles', '-c', type=int, default=2)
    parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None, help="latlon bounds: west, south, east, north")

    args=parser.parse_args()
    
    if args.verbose:
        print("working on :")
        print(args.file)
    if args.File is None:
        glob_str='%s/*ATL06*_*_%04d??%02d_*.h5' % (args.directory, args.rgt, args.subproduct)
        files=glob.glob(glob_str)
    else:
        files=[args.File]
    
    if args.pair is None:
        pairs=[1, 2, 3]
    else:
        pairs=[args.pair]
    for pair in pairs:
        #P11_list=fit_ATL11(files, N_cycles=args.cycles,  beam_pair=pair, verbose=args.verbose, first_ref_pt=args.first_point, last_ref_pt=args.last_point) # defined in ATL06_to_ATL11
        P11_list=fit_ATL11(files, N_cycles=args.cycles, beam_pair=pair, verbose=args.verbose, first_ref_pt=args.first_point, last_ref_pt=args.last_point, lonlat_bounds=args.bounds) # defined in ATL06_to_ATL11
        
        if P11_list:
            N_cycles=np.nanmax([Pi.N_cycles for Pi in P11_list])
            N_coeffs=np.nanmax([Pi.N_coeffs  for Pi in P11_list])
            ATL11.data(track_num=P11_list[0].rgt, pair_num=pair, N_cycles=N_cycles, N_coeffs=N_coeffs, N_pts=len(P11_list)).from_list(P11_list).write_to_file(args.out_file)

if __name__=="__main__":
    main(sys.argv)
