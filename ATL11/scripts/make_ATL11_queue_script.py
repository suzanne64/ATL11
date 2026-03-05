#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 2019

@author: ben
"""
import argparse
import os
import glob
import re

parser=argparse.ArgumentParser()
parser.add_argument('--list_file','-l', type=str)
parser.add_argument('--glob_06','-g', type=str, help="glob string used to find rgt / subproduct combos")
parser.add_argument('--ATL06_dir','-A', type=str, help="directory or glob string used by ATL06_to_ATL11 to find input files")
parser.add_argument('--index_glob','-i', type=str)
parser.add_argument('--out_dir','-o', type=str)
#parser.add_argument('--Version','-V', type=str, default='000')
parser.add_argument('--Version','-V', type=str, default='00')
parser.add_argument('--Release','-R', type=str, default='000')
parser.add_argument('--cycles','-c', type=str, nargs=2, default=[3, 4])
parser.add_argument('--replace', action='store_true')
parser.add_argument('--Hemisphere', '-H', type=int, help='hemisphere, -1 = Antarctic, 1=arctic', required=True)
args=parser.parse_args()

project_bin="/att/nobackup/project/icesat-2/bin"

# Greenland current best version--- user the list of ATL06 files in /home/ben/git_repos/ATL11/Greenland_ATL06_list.txt
#python3 ~/git_repos/ATL11/make_ATL11_queue.py  -H 1 -i "/Volumes/ice2/ben/scf/GL_06/003/tiles/*/GeoIndex.h5" -A "/Volumes/ice2/ben/scf/GL_06/003/cycle_*/" -l /home/ben/git_repos/ATL11/Greenland_ATL06_list.txt -o /Volumes/ice2/ben/scf/GL_11/U07 -V U07 -c 3 7 > ~/temp/ATL11_run/GL_queue.txt


re_06=re.compile('ATL06_\d+_(\d\d\d\d)\d\d(\d\d)_(R?\d\d\d)')

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

if args.glob_06 is None:
    args.glob_06 = args.ATL06_dir+'/ATL06*.h5'


ATL06_list=[]
if args.list_file is not None:
    with open(args.list_file) as fh:
        for line in fh:
            ATL06_list.append(line.rstrip())
else:
    ATL06_list=glob.glob(args.glob_06)

proc_list=[]

with open(project_bin+"/cf_atlas_l3b_is_MetaTemplate.ctl", "r") as ctltemplate:
    ctllines = ctltemplate.readlines()
for file in ATL06_list:
     
    #print(file)
    rgt, subproduct, release = re_06.search(file).groups()
    if (rgt, subproduct) in proc_list:
        continue
#    out_file="%s/ATL11_%04d%02d_%02d%02d_%02d_v%s.h5" %( \
#    out_file="%s/ATL11_%04d%02d_%02d%02d_%03d_%02d.h5" %( \
    out_file="%s/ATL11_%04d%02d_%02d%02d_%s_%s.h5" %( \
            args.out_dir, int(rgt), int(subproduct), int(args.cycles[0]), \
            int(args.cycles[1]), args.Release, args.Version)
#    out_file="%s/ATL11_%04d%02d_%02d%02d_%03d_%s.h5" %( \
#            args.out_dir, int(rgt), int(subproduct), int(args.cycles[0]), \
#            int(args.cycles[1]), int(release), args.Version)

    if os.path.isfile(out_file):
        continue

    proc_list.append((rgt, subproduct))
    #echo $out_file
#    cmd=f'python3 /home/ben/git_repos/ATL11/ATL06_to_ATL11.py {rgt} {subproduct} -o {args.out_dir} -d "{args.ATL06_dir}" -V {args.Version} -c {args.cycles[0]} {args.cycles[1]} -H {args.Hemisphere}'

    cmd=f''+os.path.join(os.path.dirname(os.path.realpath(__file__)),'run_all_l3b_is.sh')+f' --rgt {rgt} --region {subproduct} --output_path {args.out_dir} --atl06_datapath \'{args.ATL06_dir}\' --version {args.Version} --start_cycle {args.cycles[0]} --end_cycle {args.cycles[1]} --hemisphere {args.Hemisphere} --release {args.Release}'
    if args.index_glob is not None:
        cmd +=f' -G "{args.index_glob}"'
    if args.Hemisphere is 1:    
        cmd +=f' --dem_mosaic '+os.path.join(project_bin,'arcticdem_mosaic_500m_v3.0.tif')
    else:
        cmd +=f' --dem_mosaic '+os.path.join(project_bin,'mosaic_500m_dem_filled.tif')
    cmd +=f' --ctl_file {out_file}.ctl'
    print(cmd)

#    with open("/etc/apt/sources.list", "r") as ctltemplate:
#        ctllines = ctltemplate.readlines()
    with open(out_file+'.ctl', "w") as ctlfile:
        for line in ctllines:
            ctlfile.write(re.sub(r'_atl11file_', out_file, line))

#
#    cmd=f'python3 '+os.path.join(os.path.dirname(os.path.realpath(__file__)),'ATL11_browse_plots.py')+f' {out_file} -H {args.Hemisphere} -m {project_bin}/arcticdem_mosaic_500m_v3.0.tif'
#    print(cmd)

