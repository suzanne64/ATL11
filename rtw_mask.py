#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:38:56 2020

@author: ben
"""
import os
import re
import pandas as pd
import numpy as np

def read_rtw_times_from_excel(xls_file=None, to_csv=False):
    if xls_file is None:
        xls_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ICESat-2_TechRefTable_05012020.xlsx')
    rtw_rows=[]
    rtw_re = re.compile('RTWscan')
    rtw_times=[]
    for cycle in range(1, 7):
        df = pd.read_excel(xls_file, sheet_name=f'ATLAS Activities Cycle {cycle}', header=1)
        for row in range(len(df['DETAILS'])):
            if rtw_re.search(df['DETAILS'][row]):
                rtw_rows.append(row)
                rtw_times.append((df['ATL03 DELTA_TIME START (seconds)'][row], df['ATL03 DELTA_TIME END (seconds)'][row]))
    if to_csv:
        csv_file=xls_file.replace('.xlsx','_RTWs.csv')
        with open(csv_file,'w') as fh:
            for rtw_time in rtw_times:
                fh.write('%d,%d\n'% rtw_time)
    return rtw_times

def read_rtw_times_from_csv(csv_file=None):
    if csv_file is None:
        csv_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ICESat-2_TechRefTable_05012020_RTWs.csv')
    rtw_times=[]
    with open(csv_file,'r') as fh:
        for line in fh:
            rtw_times.append(tuple(map(int, line.split(','))))
    return rtw_times


def rtw_mask_for_delta_time(delta_time, rtw_times=None, csv_file=None):
    if rtw_times is None:
        rtw_times=read_rtw_times_from_csv(csv_file)
    valid=np.ones_like(delta_time, dtype=bool)
    for rtw_time in rtw_times:
        valid &= ~((delta_time>rtw_time[0]) & (delta_time < rtw_time[1]))
    return valid
