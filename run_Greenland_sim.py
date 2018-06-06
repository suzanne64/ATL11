# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:24:11 2017

@author: ben
"""
from glob import glob
import numpy as np
from ATL06_to_ATL11 import fit_ATL11
from ATL11 import ATL11_data

from ATL11_plot import ATL11_plot
#import sys
#from ATL11 import ATL11_data.write_to_file
import re, os
 
np.seterr(invalid='ignore')

#try:
#    del sys.modules['ATL06_data']; 
#except:
#    pass
#from ATL06_data import ATL06_data 
#filenames=glob('/Volumes/ice1/ben/sdt/ATLxx_example/PIG_Collab_v13A/ATL06/run_1/rep_*/Track_462_D3.h5')
ATL06_base='/Volumes/ice2/nick/34_IceSat2/SyntheticTracks_Alex/ATL06/'
track_files=[os.path.basename(X) for X  in glob(ATL06_base+'TrackData_01/*.h5')]
#filenames=glob('/Volumes/ice1/ben/sdt/ATLxx_example/PIG_Collab_v13B_NoFirn_NoDz/ATL06/run_1/rep_*/Track_462_D3.h5') 

for track_file in track_files:
    #if '0414.h5' in track_file:
    for pair in [1,2,3]:
        # establish output file name
        m=re.search(r"Track_(.*?).h5",track_file)
        fileout='Fit_%s_Pair%d.h5' % (m.group(1), pair)
        print(fileout)
        h5_files=glob(ATL06_base+'/*/'+track_file) 
        P11_list=fit_ATL11(h5_files, beam_pair=pair, seg_x_centers=None, num_centers=20, DOPLOT=None, DEBUG=False) # defined in ATL06_to_ATL11 
        if P11_list:
            D11=ATL11_data(len(P11_list), P11_list[0].N_reps).from_list(P11_list)
            #ATL11_plot(D11, P11_list)
            D11.plot()
            ATL11_data.write_to_file(D11,fileout)
        break
    
