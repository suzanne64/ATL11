#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:02:14 2019

@author: ben
"""

from PointDatabase import geo_index
import glob
import os
import sys
import argparse

def make_queue(thedir, hemisphere):
    out_dir=thedir+'/index'
    for file in glob.glob(os.path.join(thedir,'*.h5')):
        out_file=out_dir+'/'+os.path.basename(file)
        print(f"python3  /home/ben/git_repos/ATL11/index_ATL11.py -i {file} -o {out_file} -H {hemisphere}")

def get_proj4(hemisphere):
    if hemisphere==-1:
        return'+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs' 
    if hemisphere==1:
        return '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '

def make_index(in_file, out_file, hemisphere):
    SRS_proj4 = get_proj4(hemisphere)
    
    if os.path.isfile(out_file):
        os.remove(out_file)
    
    geo_index(SRS_proj4=SRS_proj4, delta=[1.e4, 1.e4]).for_file(in_file, 'ATL11').to_file(out_file)

def index_index(index_dir, hemisphere):
    SRS_proj4 = get_proj4(hemisphere) 
    index_list=[]
    for filename in glob.glob(index_dir+'/*.h5'):
        try:
            this_index=geo_index(SRS_proj4=SRS_proj4, delta=[1e4, 1e4]).for_file(filename, 'h5_geoindex', number=0)
            index_list.append(this_index)
        except Exception as e:
            print(f"-------------------problem with {filename}:")
            print(e)
            print("--------------------------------------------\n")
    geo_index(delta=[1e4, 1e4], SRS_proj4=SRS_proj4).from_list(index_list).to_file(index_dir+'/GeoIndex.h5')


parser=argparse.ArgumentParser("index an ATL11 file")
parser.add_argument('--dir','-d', type=str)
parser.add_argument('--Hemisphere','-H', type=int)
parser.add_argument('--in_file','-i', type=str)
parser.add_argument('--out_file','-o', type=str)
parser.add_argument('--make_Queue','-q')
parser.add_argument('--make_Geoindex','-G', action='store_true')
args=parser.parse_args()

if args.make_Queue is not None:
    make_queue(args.dir, args.Hemisphere)
    sys.exit(-1)

if args.make_Geoindex:
    index_index(args.dir, args.Hemisphere)
    sys.exit(-1)

make_index(args.in_file, args.out_file, args.Hemisphere)
    

