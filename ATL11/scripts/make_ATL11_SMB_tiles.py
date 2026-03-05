#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:22:51 2025

@author: ben
"""

import pointCollection as pc
import numpy as np
import re
import argparse
import sys
import os

argv=sys.argv

for i, arg in enumerate(argv):
    if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

parser=argparse.ArgumentParser(description='Collect ATL11 data for a tile.', fromfile_prefix_chars="@")
parser.add_argument('--xy0', type=float, nargs=2, help='bin center')
parser.add_argument('--width','-W', type=float, default=2.e5, help='bin size, m')
parser.add_argument('--index_dir','-i', default=os.getcwd(), help="directory in which to find ATL11 files")
parser.add_argument('--SMB_file', type=str, default=None, help="File containing SMB/FAC")
parser.add_argument('--SMB_name', type=str, help="name of SMB/FAC model")
parser.add_argument('--EPSG', type=int, default=3031, help='EPSG for coordinate system')
parser.add_argument('--decimate_bin_size', type=float, default=10000, help='size of bins used to decimate ATL11 data')
parser.add_argument('--decimate_n_target', type=int, default=1000, help='max number of points in each decimate bin')
parser.add_argument('--output_directory','-o', type=str)
parser.add_argument('--floating_mask_file', type=str)
args=parser.parse_args(argv[1:])

N_target = args.decimate_n_target
decimate_bin_size=args.decimate_bin_size
mask_file=args.floating_mask_file
SMB_FAC_file = args.SMB_file
EPSG=args.EPSG


box = [jj+args.width*np.array([-1/2, 1/2]) for jj in args.xy0]
GI_file = args.index_dir+'/GeoIndex.h5'

out_file = os.path.join(args.output_directory, f'E{int(args.xy0[0]/1.e3)}_N{int(args.xy0[1]/1.e3)}.h5')
print(out_file)

if os.path.isfile(out_file):
    D11=pc.data().from_h5(out_file)
else:

    Q11 = pc.geoIndex().from_file(GI_file).query_xy_box(*box, get_data=False)
    field_dict={None:['latitude','longitude','h_corr',
                                   'h_corr_sigma', 'h_corr_sigma_systematic',
                                   'delta_time','quality_summary', 'ref_pt'],
                    'ref_surf': ['dem_h', 'x_atc','fit_quality','e_slope','n_slope']}

    D11=[]
    count=0
    track_re=re.compile('ATL11_(\d\d\d\d)')

    for index_1 in Q11.keys():
        Q11a = pc.geoIndex().from_file(os.path.join(args.index_dir, index_1)).query_xy_box(*box, get_data=False)
        for file_sub, index_info in Q11a.items():
            file, pair = file_sub.split(':pair')
            D11i = pc.ATL11.data(pair=int(pair)).from_h5(os.path.join(args.index_dir, file),
                                                         field_dict=field_dict,
                                                         index_range = [index_info['offset_start'][0], index_info['offset_end'][0]])
            D11i.assign(pair=np.zeros_like(D11i.delta_time)+D11i.pair)
            D11i.assign(rgt = np.zeros_like(D11i.delta_time) + int(track_re.search(file).group(1)))
            D11 += [D11i]

    D11=pc.data(columns=D11[0].shape[1]).from_list(D11)
    rows=np.flatnonzero(D11.fit_quality[:,0]==0)
    D11=D11[rows,:]

    D11.get_xy(EPSG=EPSG)

    D11.assign(year=2018+D11.delta_time/24/3600/365.25)

    mask = pc.grid.data().from_geotif(mask_file, bounds=D11.bounds(pad=1.e3))
    mask.z=np.nan_to_num(mask.z)
    floating=mask.interp(D11.x[:,0], D11.y[:,0])
    print(D11.x.shape)
    D11.index(floating<0.1)
    print(D11.x.shape)
    if D11.size < 1:
        print(f"Found only {D11.size} points, exiting.")
        sys.exit()
    
    _, i10 = pc.unique_by_rows(np.round(np.c_[D11.x[:,0], D11.y[:,0]]/decimate_bin_size)*decimate_bin_size, return_dict=True)

    D11_subsamp=[]
    for ii in i10.values():
        ind = np.unique(np.round(np.linspace(0, len(ii)-1, N_target)).astype(int))
        D11_subsamp += [D11[ii[ind]]]
    D11=pc.data(columns = D11.columns).from_list(D11_subsamp)

    D11.to_h5(out_file, replace=True)

if args.SMB_name=='CFM_1d':
    # CFM_1d has time in days since 2018
    SMB = pc.grid.data().from_nc(SMB_FAC_file, bounds=D11.bounds(pad=5.e4))
    SMB.t = 2018+SMB.t/365.25
else:
    SMB = pc.grid.data().from_nc(SMB_FAC_file, bounds=D11.bounds(pad=5.e4), t_range=[np.nanmin(D11.year), np.nanmax(D11.year)])
SMB.FAC=pc.grid.fill_edges(SMB.FAC)
SMB.SMB_a=pc.grid.fill_edges(SMB.SMB_a)

SMB_FAC={}
for field in ['FAC','SMB_a']:
    SMB_FAC[field]=SMB.interp(D11.x, D11.y, D11.year, field=field)

pc.data().from_dict(SMB_FAC).to_h5(out_file, group=args.SMB_name, replace=False)

