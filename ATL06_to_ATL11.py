
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

def buffered_bins(x0, y0, w_buffer, w_bin, complex=True):
    """
    Generate a set of bins that enclose a set of points, including a buffer
    """
    dx, dy=np.meshgrid([-w_buffer, 0, w_buffer], [-w_buffer, 0, w_buffer])
    dx.shape=[9, 1];
    dy.shape=[9, 1]
    xr=np.unique(np.round(x0/w_buffer)*w_buffer+1j*np.round(y0/w_buffer)*w_buffer)
    xr=np.unique(xr.ravel()+dx+1j*dy)
    xr=np.unique(np.round(xr/w_bin)*w_bin)
    if complex:
        return xr
    else:
        return np.real(xr), np.imag(xr)

def unwrap_lon(lon, lon0=0):
    """
    wrap longitudes to +-180
    """
    lon -= lon0
    lon[lon>180] -= 360
    lon[lon<-180] += 360
    return lon

def get_xover_data(x0, y0, rgt, GI_file, xover_cache, index_bin_size, params_11):
    """
    Read the data from other tracks.

    Maintain a cache of data so that subsequent reads don't have to reload data from disk
    Inputs:
        x0, y0: bin centers
        rgt: current rgt
        GI_file: geograpic index file
        xover_cache: data cache (dict)
        index_bin_size: size of the bins in the index
        params_11: default parameter values for the ATL11 fit

    """

    # identify the crossover centers
    x0_ctrs=buffered_bins(x0, y0, 2*params_11.L_search_XT, index_bin_size)
    D_xover=[]

    for x0_ctr in x0_ctrs:
        this_key=(np.real(x0_ctr), np.imag(x0_ctr))
        # check if we have already read in the data for this bin
        if this_key not in xover_cache:
            # if we haven't already read in the data, read it in.  These data will be in xover_cache[this_key]
            xover_cache[this_key]={'D':point_data(field_dict=params_11.ATL06_field_dict).from_list(geo_index().from_file(GI_file).query_xy(this_key, fields=params_11.ATL06_field_dict))}
            # remove the current rgt from data in the cache
            xover_cache[this_key]['D'].index(~np.in1d(xover_cache[this_key]['D'].rgt, [rgt]))
            if xover_cache[this_key]['D'].size==0:
                continue
            xover_cache[this_key]['D'].get_xy(EPSG=params_11.EPSG)
            # index the cache at 100-m resolution
            xover_cache[this_key]['index']=geo_index(delta=[100, 100], data=xover_cache[this_key]['D'])
        # now read the data from the crossover cache
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


def regress_to(D, out_field_names, in_field_names, in_field_pt, DEBUG=None):

    """
    Regress a series of data values to a reference point

    inputs:
        D: data structure
        out_field_names: list of field names for which we are trying to recover values
        in_field_names: list of field names as independent variables in the regression
        in_field_pt: location of the regression center (in variables in_field_names)
    """

    D_in = np.array((getattr(D, in_field_names[0]).ravel(), getattr(D, in_field_names[1]).ravel())).T
    D_out = np.array([getattr(D,ff).ravel() for ff in out_field_names]).T
    if ['longitude'] in out_field_names:
        # if longitude is in the regression parameters, need to unwrqp it
        lon_col=out_field_names.index['longitude']
        lon0=np.nanmedian(D_out[lon_col])
        D_out[:, lon_col]=unwrap_lon(D_out[:, lon_col], lon0=lon0)
    good_rows=np.all(~np.isnan( np.concatenate( (D_in,D_out), axis=1)), axis=1)

    # build the regression matrix
    G=np.ones((np.sum(good_rows),len(in_field_names)+1) )
    for k in range(len(in_field_names)):
        G[:,k+1] = D_in[:,k] - in_field_pt[k]

    out_pt = np.linalg.lstsq(G,D_out[good_rows,:])[0]
    if ['longitude'] in out_field_names:
        out_pt[lon_col] = unwrap_lon([out_pt[lon_col]+lon0], lon0=0)[0]
    if DEBUG is not None:
        print('lat_ctr,lon_ctr',out_pt)

    return out_pt

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
        D11==ATL06_data().from_ATL06(files, N_cycles=args.cycles, beam_pair=pair, verbose=args.verbose, first_ref_pt=args.first_point, last_ref_pt=args.last_point, lonlat_bounds=args.bounds) # defined in ATL06_to_ATL11

        D11.write_to_file(args.out_file)

if __name__=="__main__":
    main(sys.argv)
