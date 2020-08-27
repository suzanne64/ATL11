#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:38:56 2020

@author: ben
"""
import os
import re
import numpy as np

def read_rtw_from_excel(xls_file=None, to_csv=False):
    import pandas as pd
    if xls_file is None:
        xls_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ICESat-2_TechRefTable_05012020_rgts.xlsx')
    rtw_rows=[]
    rtw_re = re.compile('RTWscan')
    rtw_times=[]
    rtw_orb=[]
    rtw_rgt=[]
    rtw_cycle=[]
    for cycle in range(1, 8):
        print(f'cycle= {cycle}')
        df = pd.read_excel(xls_file, sheet_name=f'ATLAS Activities Cycle {cycle}', header=1)
        for row in range(len(df['DETAILS'])):
            if rtw_re.search(df['DETAILS'][row]):
                rtw_rows.append(row)
                rtw_times.append((df['ATL03 DELTA_TIME START (seconds)'][row], df['ATL03 DELTA_TIME END (seconds)'][row]))
                rtw_orb.append((df['BEG ORBIT'][row], df['END ORBIT'][row]))
                rtw_rgt.append((df['BEG RGT'][row], df['END RGT'][row]))
                rtw_cycle.append(cycle)
    if to_csv:
        print('writing!')
        csv_file=xls_file.replace('.xlsx','_RTWs.csv')
        with open(csv_file,'w') as fh:
            fh.write('delta_time0, delta_time1, orb0, orb1, rgt0, rgt1, cycle\n')
            for ii, rtw_time in enumerate(rtw_times):
                fh.write('%d,%d,%4.0f,%4.0f,%4.0f,%4.0f,%4.0f\n'% (rtw_time[0], rtw_time[1], rtw_orb[ii][0], rtw_orb[ii][1], rtw_rgt[ii][0],rtw_rgt[ii][1], rtw_cycle[ii]))
    return rtw_times, rtw_orb, rtw_rgt, rtw_cycle

def read_rtw_from_csv(csv_file=None):
    if csv_file is None:
        csv_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ICESat-2_TechRefTable_07222020_RTWs.csv')
    rtw_times=[]
    rtw_orb=[]
    rtw_rgt=[]
    rtw_cycle=[]
    with open(csv_file,'r') as fh:
        for line in fh:
            try:
                temp=list(map(int, line.rstrip().split(',')))
                rtw_times.append((temp[0], temp[1]))
                rtw_orb.append((temp[2], temp[3]))
                rtw_rgt.append((temp[4], temp[5]))
                rtw_cycle.append(temp[6])
            except Exception as e:
                pass
    return rtw_times, rtw_orb, rtw_rgt, rtw_cycle


def rtw_mask_for_delta_time(delta_time, rtw_times=None, csv_file=None):
    if rtw_times is None:
        rtw_times=read_rtw_from_csv(csv_file)[0]
    valid=np.ones_like(delta_time, dtype=bool)
    for rtw_time in rtw_times:
        valid &= ~((delta_time>rtw_time[0]) & (delta_time < rtw_time[1]))
    return valid

def rtw_mask_for_orbit(orb, rtw_times=None, csv_file=None):
    if rtw_times is None:
        rtw_orbs=read_rtw_from_csv(csv_file)[1]
    valid=np.ones_like(orb, dtype=bool)
    for rtw_orb in rtw_orbs:
        valid &= ~((orb>=rtw_orb[0]) & (orb <= rtw_orb[1]))
    return valid