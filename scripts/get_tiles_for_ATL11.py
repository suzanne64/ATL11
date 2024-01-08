#! /usr/bin/env python


import pointCollection as pc
import numpy as np
import glob
import sys
import os
from argparse import ArgumentParser

def get_ATL11_xy(ATL11_file, hemisphere):
    # read lat/lon from ATL11 file:
    D0=[]
    D0 = [pc.data().from_h5(ATL11_file, group=pt, fields=['latitude','longitude']) for  pt in ['pt1','pt2','pt3']]
    D0=pc.data().from_list(D0)
    if hemisphere==-1:
        D0.get_xy(EPSG=3031)
    else:
        D0.get_xy(EPSG=3413)
    # get unique 10-km bin locations:
    u_xy = pc.unique_by_rows(np.round(np.c_[D0.x, D0.y]/1.e4)*1.e4)

    return u_xy

def tiles_for_xy(u_xy, GI_files):
    
    tile_files=[]

    for GI_file in GI_files:
        # query the geoIndex file
        Q1=pc.geoIndex().from_file(GI_file, read_file=False).query_xy([u_xy[:,0], u_xy[:,1]], pad=1, get_data=False)
        if Q1 is not None:
            for tile_file in Q1.keys():
                if os.path.isfile(tile_file):
                    tile_files += tile_file
                else:
                    tile_files += [os.path.join(os.path.dirname(GI_file), tile_file)]
    return tile_files


def __main__():
    parser=ArgumentParser('Find tile files needed to make an ATL11 file')
    parser.add_argument('--hemisphere', type=int, default=-1, help='hemisphere, specify -1 for Antarctic or 1 for Arctic')
    parser.add_argument('ATL11_file', type=str, help="ATL11 file from which to read lat/lon data")
    parser.add_argument('GeoIndex_files', type=str, nargs='+', help="GeoIndex files to query")
    args=parser.parse_args()
    
    u_xy = get_ATL11_xy(args.ATL11_file, args.hemisphere)
    
    tile_files = tiles_for_xy(u_xy, args.GeoIndex_files)
    
    print(' '.join(tile_files))

if __name__=='__main__':
    __main__()
    
