#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:52:39 2020

@author: root
"""
import os, h5py
import numpy as np
import ATL11

def write_METADATA(outfile,infiles):
    if os.path.isfile(outfile):        
        g = h5py.File(outfile,'r+')
        g.create_group('METADATA')
        gl = g['METADATA'].create_group('Lineage')
        gf = gl.create_group('ATL06')
        fname = []
        sname = []
        scycle = np.array([13],dtype='int32')
        ecycle = np.array([0],dtype='int32')
        srgt = np.array([1400],dtype='int32')
        ergt = np.array([0],dtype='int32')
        sregion = np.array([15],dtype='int32')
        eregion = np.array([0],dtype='int32')
        sorbit = []
        eorbit = []
        uuid = []
        version = []
        for ii,infile in enumerate(sorted(infiles)):
            fname.append(os.path.basename(infile).encode('ASCII'))
            sname.append('_'.join(os.path.basename(infile).split('_',2)[:2]).encode('ASCII'))
            digits =infile.split('ATL06_')[1].split('_')
            print('digits',digits)
            if scycle > np.int32(digits[1][4:6]):
                scycle = np.int32(digits[1][4:6])
            if ecycle < np.int32(digits[1][4:6]):
                ecycle = np.int32(digits[1][4:6])
            if srgt > np.int32(digits[1][:4]):
                srgt = np.int32(digits[1][:4])
            if ergt < np.int32(digits[1][:4]):
                ergt = np.int32(digits[1][:4])
            if sregion > np.int32(digits[1][6:8]):
                sregion = np.int32(digits[1][6:8])
            if eregion < np.int32(digits[1][6:8]):
                eregion = np.int32(digits[1][6:8])
            if os.path.isfile(infile):
                f = h5py.File(infile,'r')
                sorbit.append(f['METADATA']['Lineage']['ATL03'].attrs['start_orbit'])
                eorbit.append(f['METADATA']['Lineage']['ATL03'].attrs['end_orbit'])
                uuid.append(f['METADATA']['Lineage']['ATL03'].attrs['uuid'])
            version.append(str(digits[2]).encode('ASCII'))

        gf.attrs['description'] = 'ICESat-2 ATLAS Land Ice'
        gf.attrs['fileName'] = fname
        gf.attrs['shortName'] = sname
        
        gf.attrs['start_orbit'] = sorbit
        gf.attrs['end_orbit'] = eorbit
        
        gf.attrs['start_cycle'] = scycle
        gf.attrs['end_cycle']   = ecycle
        
        gf.attrs['start_rgt'] = srgt
        gf.attrs['end_rgt']   = ergt

        gf.attrs['start_region'] = sregion
        gf.attrs['end_region'] = eregion
        
        try:
            gf.attrs['start_geoseg'] = np.min(g['pt1']['corrected_h']['ref_pt'][:])
            gf.attrs['end_geoseg']   = np.max(g['pt1']['corrected_h']['ref_pt'][:])
        except IOError as e:
            print('Could not read ref_pt min/max'.format(e))
        else:
            gf.attrs['start_geoseg'] = np.min(g['pt2']['corrected_h']['ref_pt'][:])
            gf.attrs['end_geoseg']   = np.max(g['pt2']['corrected_h']['ref_pt'][:])
        finally:
            gf.attrs['start_geoseg'] = np.min(g['pt3']['corrected_h']['ref_pt'][:])
            gf.attrs['end_geoseg']   = np.max(g['pt3']['corrected_h']['ref_pt'][:])
           
        
        gf.attrs['uuid'] = uuid
        gf.attrs['version'] = version

        g.close()
    return outfile    
    
#if __name__=='__main__':
#    outfile = write_METADATA(outfile,infiles)
    
    
    