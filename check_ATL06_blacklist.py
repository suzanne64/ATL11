# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:26:47 2019

@author: ben
"""
import os
import glob
import re
import numpy as np

def check_ATL06_blacklist(filename, blacklist=None, blacklist_dir='/Volumes/ice2/ben/scf/Blacklists/'):
    if blacklist is None:
        file_re=re.compile('(ATL06\S+.h5)')
        blacklist=list()
        for file in glob.glob(blacklist_dir+'/*.txt'):
            with open(file,'r') as fh:
                for line in fh:
                    m=file_re.search(line)
                    if m is not None:
                        blacklist.append(m.group(1))
    return os.path.basename(filename.split(':')[0]) in blacklist, blacklist
    
def check_rgt_cycle_blacklist(filename=None, rgt_cycle=None, blacklist=None, blacklist_files=None):
    if blacklist is None and blacklist_files is None:
        script_dir=os.path.dirname(os.path.realpath(__file__))
        blacklist_files=[ 'blacklists/Combined_rel001_rel002_Sep_2019.csv', 'blacklists/rel002_Nov_2019.csv']
        blacklist_files=[os.path.join(script_dir, item) for item in blacklist_files]
    if blacklist is None:
        if blacklist_files is not None:
            blacklist={'rgt':[], 'cycle':[]}
            for blacklist_file in blacklist_files:
                with open(blacklist_file,'r') as bfh:
                    for line in bfh:
                        try:
                            line=line.replace(',',' ')
                            items=line.split()
                            blacklist['rgt'].append(int(items[0]))
                            blacklist['cycle'].append(int(items[1]))
                        except ValueError:
                            continue
            blacklist=[item for item in zip(blacklist['rgt'], blacklist['cycle'])]
    if filename is not None:
        r06=re.compile('ATL.._(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)_(\d\d\d\d)(\d\d)(\d\d)_(\d\d\d)_(\d\d).h5')
        m=r06.search(filename)
        if m is None:
            raise ValueError("filename does not match the template")
        return (int(m.group(7)), int(m.group(8))) in blacklist, blacklist
    elif rgt_cycle is not None:
        if hasattr(rgt_cycle[0],'__iter__'):
            result = [tuple(zz) in blacklist for zz in zip(rgt_cycle[0], rgt_cycle[1])]
        else:
            result=tuple(rgt_cycle) in blacklist
        return result, blacklist
    else:
        return blacklist


