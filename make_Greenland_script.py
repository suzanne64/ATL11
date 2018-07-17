# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:24:11 2017

@author: ben
"""
from glob import glob
import numpy as np
import re, os
 
# script that generates a list of command line calls that can be passed to gnu parallel 
 
# np.seterr(all=raise)
np.seterr(invalid='ignore')

ATL06_base='/Volumes/ice2/nick/34_IceSat2/SyntheticTracks_Alex/ATL06/'
# Make a list of the track files in the first cycle:
track_file_list=[os.path.basename(X) for X  in glob(ATL06_base+'TrackData_01/*.h5')]
# loop over the tracks: 
for track_file in track_file_list:
    # loop over pairs
    for pair in [1,2,3]:
        # establish output file name
        m=re.search(r"Track_(.*?).h5",track_file)
        track_num=int(m.group(1))
        fileout='ATL11_Track%s_Pair%d.h5' % (track_num, pair)
        #print fileout
        if os.path.isfile(fileout):
            continue
        glob_str="%s/TrackData_*/Track_%s.h5" % (ATL06_base, m.group(1))
        print "python ATL06_to_ATL11.py --ATL06_glob '%s' -o %s -v -p %d -t %d" %(glob_str, fileout, pair, track_num)
        
      