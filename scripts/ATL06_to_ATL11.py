#! /usr/bin/env python3 -u

'''
Executable script to generate ATL11 files based on ATL06 data.
'''

import os
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
os.environ['OPENBLAS_NUM_THREADS']="1"

import numpy as np
import ATL11
#import write_METADATA
import time
import glob
import sys
import matplotlib.pyplot as plt
import resource as memresource


#591 10 -F /Volumes/ice2/ben/scf/AA_06/001/cycle_02/ATL06_20190205041106_05910210_001_01.h5 -b -101. -76. -90. -74.5 -o test.h5 -G "/Volumes/ice2/ben/scf/AA_06/001/cycle*/index/GeoIndex.h5"
#591 10 -F /Volumes/ice2/ben/scf/AA_06/001/cycle_02/ATL06_20190205041106_05910210_001_01.h5 -o test.h5 -G "/Volumes/ice2/ben/scf/AA_06/001/cycle*/index/GeoIndex.h5" 

def get_proj4(hemisphere):
    if hemisphere==-1:
        return'+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs' 
    if hemisphere==1:
        return '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '

def main(argv):
    # Tunable: 2000 sounds OK
    BLOCKSIZE = 2000
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    # command-line interface: run ATL06_to_ATL11 on a list of ATL06 files
    import argparse
    parser=argparse.ArgumentParser(description='generate an ATL11 file from a collection of ATL06 files.')
    parser.add_argument('rgt', type=int, help="reference ground track number")
    parser.add_argument('subproduct', type=int, help="ICESat-2 subproduct number (latltude band)")
    parser.add_argument('--directory','-d', default=os.getcwd(), help="directory in which to search for ATL06 files")
    parser.add_argument('--pair','-p', type=int, default=None, help="pair number to process (default is all three)")
    parser.add_argument('--Release','-R', type=int, default=2, help="Release number")
    parser.add_argument('--Version','-V', type=int, default=1, help="Version number")
    parser.add_argument('--cycles', '-c', type=int, nargs=2, default=[3, 4], help="first and last cycles")
    parser.add_argument('--GI_file_glob','-G', type=str, default=None, help="Glob (wildcard) string used to match geoindex files for crossing tracks")
    parser.add_argument('--out_dir','-o', default=None, required=True, help="Output directory")
    parser.add_argument('--first_point','-f', type=int, default=None, help="First reference point")
    parser.add_argument('--last_point','-l', type=int, default=None, help="Last reference point")
    parser.add_argument('--num_points','-N', type=int, default=None, help="Number of reference points to process")
    parser.add_argument('--Hemisphere','-H', type=int, default=-1)
    parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None, help="latlon bounds: west, south, east, north")
    parser.add_argument('--max_xover_latitude', type=float, default=90, help="highest latitude for which crossovers will be calculated")
    parser.add_argument('--test_plot', action='store_true', help="plots locations, elevations, and elevation differences between cycles")
    parser.add_argument('--xy_bias_file', type=str, help="CSV file containing fields delta_time, x_bias, and y_bias")
    parser.add_argument('--Blacklist','-B', action='store_true')
    parser.add_argument('--verbose','-v', action='store_true')
    args=parser.parse_args()

    # output file format is ATL11_RgtSubprod_c1c2_rel_vVer.h5
    out_file="%s/ATL11_%04d%02d_%02d%02d_%03d_%02d.h5" %( \
            args.out_dir,args.rgt, args.subproduct, args.cycles[0], \
            args.cycles[1], args.Release, args.Version)
    if os.path.isfile(out_file):
        os.remove(out_file)

    if args.verbose:
        print('ATL11 output filename',out_file)
    glob_str='%s/*ATL06*_*_%04d??%02d_*.h5' % (args.directory, args.rgt, args.subproduct)
    files=glob.glob(glob_str)
    if args.verbose:
        print("found ATL06 files:" + str(files))

    if args.pair is None:
        pairs=[1, 2, 3]
    else:
        pairs=[args.pair]
    
    if args.GI_file_glob is not None:
        GI_files=glob.glob(args.GI_file_glob)
    else:
        GI_files=None   
    if args.verbose:
        print("found GI files:"+str(GI_files))
    
    for pair in pairs:
        # read the lat, lon, segment_id data for each segment
        D6_segdata = ATL11.read_ATL06_data(files, beam_pair=pair, 
                                           cycles=args.cycles, 
                                           use_blacklist=args.Blacklist, 
                                           minimal=True)
        all_ref_pts=[]
        all_ref_pt_x=[]
        for filename in D6_segdata.keys():
            _, ref_pt_numbers, ref_pt_x = ATL11.select_ATL06_data(\
                            D6_segdata[filename].copy(), \
                            first_ref_pt=args.first_point,\
                            last_ref_pt=args.last_point, \
                            lonlat_bounds=args.bounds,\
                            num_ref_pts=args.num_points)
            if ref_pt_numbers is None:
                continue
            all_ref_pts += [ref_pt_numbers]
            all_ref_pt_x += [ref_pt_x]
        all_ref_pts, ind =np.unique(np.concatenate(all_ref_pts), return_index=True)
        all_ref_pt_x = np.concatenate(all_ref_pt_x)[ind]
        
        # loop over all segments in blocks of BLOCKSIZE
        blocks=np.arange(0, len(all_ref_pts), BLOCKSIZE)
        D11=[]
        last_time=time.time()
        
        atc_shift_table=None
        
        for block0 in blocks:
            ref_pt_range = [all_ref_pts[block0], all_ref_pts[np.minimum(len(all_ref_pts)-1, block0+BLOCKSIZE)]]
            print(f'ref_pt_range={ref_pt_range}')
            seg_range=[np.maximum(0, ref_pt_range[0]-ATL11.defaults().N_search),
                       ref_pt_range[1]+ATL11.defaults().N_search]
            
            D6 = ATL11.read_ATL06_data(files, beam_pair=pair,
                                       cycles=args.cycles,
                                       use_blacklist=args.Blacklist,
                                       ATL06_dict=D6_segdata, seg_range = seg_range )
            
            atc_shift_table = ATL11.calc_geoloc_bias(D6, args.xy_bias_file, atc_shift_table=atc_shift_table)
            
            if D6 is None:
                continue
            #D6, ref_pt_numbers, ref_pt_x = ATL11.select_ATL06_data(D6, \
            ##                    first_ref_pt=args.first_point,\
            #                    last_ref_pt=args.last_point, \
            #                    lonlat_bounds=args.bounds, 
            #                    num_ref_pts=args.num_points)

            ref_pt_ind=(all_ref_pts >= ref_pt_range[0]) & \
                                   (all_ref_pts <= ref_pt_range[1])
            ref_pt_numbers=all_ref_pts[ref_pt_ind]
            ref_pt_x = all_ref_pt_x[ref_pt_ind]
        
            if len(ref_pt_numbers)==0: 
                continue
            D11 +=ATL11.data().from_ATL06(D6, ref_pt_numbers=ref_pt_numbers, ref_pt_x=ref_pt_x,\
                                           cycles=args.cycles, \
                                           beam_pair=pair, \
                                           verbose=args.verbose, \
                                           GI_files=GI_files, \
                                           hemisphere=args.Hemisphere, \
                                           max_xover_latitude=args.max_xover_latitude, return_list=True) # defined in ATL06_to_ATL11
            
            print("completed %d/%d blocks, ref_pt = %d, last %d segments in %2.2f s." %(list(blocks).index(block0)+1, len(blocks), np.nanmax(D6.segment_id), BLOCKSIZE, time.time()-last_time))
            print(f"memory: {memresource.getrusage(memresource.RUSAGE_SELF).ru_maxrss}")
            last_time=time.time()
        if len(D11) > 0:
            cycles=[np.nanmin([Pi.cycles for Pi in D11]), np.nanmax([Pi.cycles for Pi in D11])]
            N_coeffs=np.nanmax([Pi.N_coeffs  for Pi in D11])
            D11=ATL11.data(track_num=D11[0].rgt, beam_pair=pair, cycles=cycles, N_coeffs=N_coeffs).from_list(D11)
        else:
            D11=None
        
        if D11 is None:
            print(f"ATL06_to_ATL11: Not enough good data to calculate an ATL11 for {pair}, nothing written")
            continue
        # fill cycle_number list in cycle_stats and ROOT
        setattr(D11.cycle_stats,'cycle_number',list(range(args.cycles[0],args.cycles[1]+1)))
        setattr(D11.ROOT,'cycle_number',list(range(args.cycles[0],args.cycles[1]+1)))
        # add dimensions to D11
        D11.N_pts, D11.N_cycles = D11.ROOT.h_corr.shape
        
        if isinstance(D11.crossing_track_data.h_corr, np.ndarray):
            D11.Nxo = D11.crossing_track_data.h_corr.shape[0]
        
        if D11 is not None:
            D11.write_to_file(out_file)

#    out_file = ATL11.write_METADATA.write_METADATA(out_file,files)
    out_file = ATL11.write_METADATA(out_file,files)

    print("ATL06_to_ATL11: done with "+out_file)
        
    if args.test_plot:
        ATL11.ATL11_test_plot.ATL11_test_plot(out_file)
#        ATL11.ATL11_browse_plots.ATL11_browse_plots(out_file,args.Hemispher,mosaic=mosaic)

if __name__=="__main__":
    main(sys.argv)
