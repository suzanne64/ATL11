import numpy as np
from ATL06_data import ATL06_data
from ATL11_point import ATL11_point
from ATL11_data import ATL11_data
from ATL11_misc import ATL11_defaults
from glob import glob

def fit_ATL11(ATL06_files, beam_pair=1, ref_pt_numbers=None, output_file=None, num_ref_pts=None, first_ref_pt=None, DOPLOT=None, DEBUG=None, mission_time_bds=None, verbose=False):
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

    if first_ref_pt is not None:
        these=ref_pt_numbers>=first_ref_pt
        ref_pt_numbers=ref_pt_numbers[these]
        ref_pt_x=ref_pt_x[these]

    if num_ref_pts is not None:
        ref_pt_numbers=ref_pt_numbers[0:int(num_ref_pts)]
        ref_pt_x=ref_pt_x[0:int(num_ref_pts)]

    for count, ref_pt_number in enumerate(ref_pt_numbers):
        #print("ref_pt_number=%d" % ref_pt_number)
        x_atc_ctr=ref_pt_x[count]
        try:
            # section 5.1.1
            D6_sub=D6.subset(np.any(np.abs(D6.segment_id-ref_pt_number) <= params_11.N_search, axis=1), by_row=True)
            if D6_sub.h_li.shape[0]<=1:
                if verbose:
                    print("not enough data at ref pt=%d" % ref_pt_number)
                continue

            #2a. define representative x and y values for the pairs
            pair_data=D6_sub.get_pairs(datasets=['x_atc','y_atc','delta_time','dh_fit_dx','dh_fit_dy','segment_id','cycle','h_li'])   # this might go, similar to D6_sub
            P11=ATL11_point(N_pairs=len(pair_data.x), ref_pt_number=ref_pt_number, x_atc_ctr=x_atc_ctr, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),N_cycles=len(ATL06_files),  mission_time_bds=mission_time_bds )

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

            P11.corrected_h.ref_pt_lat,P11.corrected_h.ref_pt_lon = regress_to(D6_sub,['latitude','longitude'], ['x_atc','y_atc'],[x_atc_ctr,P11.y_atc_ctr])

            P11.find_reference_surface(D6_sub)
            if 'inversion failed' in P11.status:
                P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))

                continue

            P11.corr_heights_other_cycles(D6_sub)

            P11_list.append(P11)
        except:
            print("uncaught exception for ref_pt=%d" % ref_pt_number)
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

def main():
    # command-line interface: run ATL06_to_ATL11 on a list of ATL06 files
    import argparse
    parser=argparse.ArgumentParser(description='generate an ATL11 file from a collection of ATL06 files.')
    parser.add_argument('--verbose','-v', action='store_true')
    parser.add_argument('--pair','-p', type=int, required=True)
    parser.add_argument('--track','-t', type=int, required=True)
    parser.add_argument('--ATL06_glob','-A', default=None, required=True)
    parser.add_argument('--out_file','-o', default=None, required=True)
    args=parser.parse_args()
    ATL06_files=glob(args.ATL06_glob)
    if args.verbose:
        print("working on :")
        print(ATL06_files)
    if len(ATL06_files) <1:
        print("no files found for %s" % args.ATL06_glob)
        exit()

    P11_list=fit_ATL11(ATL06_files, beam_pair=args.pair, verbose=args.verbose) # defined in ATL06_to_ATL11
    if P11_list:
        ATL11_data(track_num=args.track, pair_num=args.pair).from_list(P11_list).write_to_file(args.out_file)

if __name__=="__main__":
    main()
