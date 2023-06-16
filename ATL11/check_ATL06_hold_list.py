# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:26:47 2019

@author: ben
"""
import os
import glob
import re
import numpy as np
from importlib import resources

def read_hold_files(hold_dir=None):
    if hold_dir is None:
        hold_dir = str(resources.files('ATL11').joinpath("package_data/held_granules"))
    rgt=[]; cycle=[]; subprod=[];
    hold_files=glob.glob(os.path.join(hold_dir, '*.csv'))
    if hold_files is None or len(hold_files)==0:
        return None
    for file in hold_files:
        with open(file,'r') as ff:
            for line in ff:
                try:
                     items=line.replace(',',' ').split()
                     cycle.append(int(items[0]))
                     rgt.append(int(items[1]))
                     subprod.append(int(items[2]))
                except ValueError:
                    continue
    hold_list=[item for item in zip(cycle, rgt, subprod)]
    return hold_list


def check_ATL06_hold_list(filenames, hold_list=None, hold_dir=None):
    if hold_list is None:
        hold_list=read_hold_files(hold_dir=hold_dir)
    if isinstance(filenames, (str)):
        filenames=list(filenames)

    r06=re.compile('ATL.._(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)_(\d\d\d\d)(\d\d)(\d\d)_(\d\d\d)_(\d\d).h5')
    bad=[]
    for filename in filenames:
        m=r06.search(filename)
        bad.append((int(m.group(8)), int(m.group(7)), int(m.group(9))) in hold_list)
    return bad

def check_ATL06_data_against_hold_list(D, hold_list=None):
    """
    Remove held rgt/cycle combinations from a data object.

    Parameters
    ----------
    D : pc.data
        ATL06 data, must have cycle (or cycle_number) and rgt fields
    hold_list : list, optinal
        list of cycle, rgt combinations that will be excluded from
        analysis. The default is None.

    Returns
    -------
    None.

    """

    if hold_list is None:
        return

    # assume that hold_list is cycle, rgt, subproduct
    if 'cycle_number' in D.fields:
        c=D.cycle_number
    else:
        c=D.cycle
    hold_arr=np.c_[hold_list]
    cr = np.round(c).astype(int)+1j*np.round(D.rgt).astype(int)
    good = ~np.in1d(cr,  hold_arr[:,0]+1j*hold_arr[:,1])
    if good.ndim==2:
        good=np.all(good, axis=2)
    if not np.all(good):
        D.index(good)
