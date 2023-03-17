#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:52:39 2020

@author: root

"""
import os, h5py
import numpy as np
import sys
import uuid
import pyproj
import shapely
import shapely.geometry
from osgeo import osr
from datetime import datetime, timedelta
#from shapely.ops import cascaded_union
from shapely.ops import unary_union
from shapely.validation import explain_validity
import ATL11
from ATL11.h5util import create_attribute, duplicate_group
from ATL11.version import softwareVersion,softwareDate,softwareTitle,identifier,series_version
from scipy.spatial import ConvexHull
import pkg_resources
# import importlib.resources

def write_METADATA(outfile,sec_offset,start_date,infiles):
    if os.path.isfile(outfile):        
#
# Call filemeta, copies METADATA group from template
#
        filemeta(outfile,sec_offset,start_date,infiles)
        g = h5py.File(outfile,'r+')
        gf = g.create_group('METADATA/Lineage/ATL06'.encode('ASCII','replace'))
        fname = []
        sname = []
        scycle = []
        ecycle = []
        srgt = []
        ergt = []
        sregion = []
        eregion = []
        sgeoseg = 0
        egeoseg = 0
        sorbit = []
        eorbit = []
        uuid = []
        version = []
        for ii,infile in enumerate(sorted(infiles)):
            fname.append(os.path.basename(infile).encode('ASCII'))
            if os.path.isfile(infile):
                f = h5py.File(infile,'r')
# Read the datasets from ATL06 ancillary_data, where available
# All fields must be arrays, not just min/max, even if just repeats
#
                sname.append(f['/'].attrs['short_name'])
                uuid.append(f['/'].attrs['identifier_file_uuid'])
                scycle.append(f['ancillary_data/start_cycle'])
                ecycle.append(f['ancillary_data/end_cycle'])
                sorbit.append(f['ancillary_data/start_orbit'])
                eorbit.append(f['ancillary_data/end_orbit'])
                sregion.append(f['ancillary_data/start_region'])
                eregion.append(f['ancillary_data/end_region'])
                srgt.append(f['ancillary_data/start_rgt'])
                ergt.append(f['ancillary_data/end_rgt'])
                version.append(f['ancillary_data/version'])
        for pt in g.keys():
            if pt.startswith('pt'):
                if sgeoseg == 0:
                    sgeoseg=np.min(g[pt]['ref_pt'][:])
                    egeoseg=np.max(g[pt]['ref_pt'][:])
                else:
                    sgeoseg = np.min([sgeoseg,np.min(g[pt]['ref_pt'][:])])
                    egeoseg = np.max([egeoseg,np.max(g[pt]['ref_pt'][:])])

        gf.attrs['description'] = 'ICESat-2 ATLAS Land Ice'.encode('ASCII','replace')
#
# Use create_attribute for strings to get ASCII and NULLTERM
#
        create_attribute(gf.id, 'fileName', [2], fname)
        create_attribute(gf.id, 'shortName', [2], sname)
        
        gf.attrs['start_orbit'] = np.ravel(sorbit)
        gf.attrs['end_orbit'] = np.ravel(eorbit)
        
        gf.attrs['start_cycle'] = np.ravel(scycle)
        gf.attrs['end_cycle']   = np.ravel(ecycle)
        
        gf.attrs['start_rgt'] = np.ravel(srgt)
        gf.attrs['end_rgt']   = np.ravel(ergt)

        gf.attrs['start_region'] = np.ravel(sregion)
        gf.attrs['end_region'] = np.ravel(eregion)
        
        gf.attrs['start_geoseg'] = np.repeat(sgeoseg,np.size(sregion))
        gf.attrs['end_geoseg'] = np.repeat(egeoseg,np.size(sregion))
                
        create_attribute(gf.id, 'uuid', [2], uuid)
        gf.attrs['version'] = np.ravel(version)

        g.close()
    return outfile    
    
#if __name__=='__main__':
#    outfile = write_METADATA(outfile,infiles)
    
def filemeta(outfile,sec_offset,start_date,infiles):

    orbit_info={'crossing_time':0., 'cycle_number':0, 'lan':0., \
        'orbit_number':0., 'rgt':0, 'sc_orient':0, 'sc_orient_time':0.}
    root_info={'asas_release':'', 'Conventions':'', 'contributor_name':'', 'contributor_role':'', 'date_created':'', 'date_type':'', \
        'description':'', 'featureType':'', 'geospatial_lat_max':0., 'geospatial_lat_min':0., \
        'geospatial_lat_units':'', \
        'geospatial_lon_max':0., 'geospatial_lon_min':0., 'geospatial_lon_units':'', 'granule_type':'', \
        'hdfversion':'', 'history':'', \
        'identifier_file_uuid':'', 'identifier_product_doi_authority':'', 'identifier_product_format_version':'', \
        'level':'', 'license':'', 'naming_authority':'', 'spatial_coverage_type':'', 'standard_name_vocabulary':'', \
        'time_coverage_duration':0., \
        'time_coverage_end':'', 'time_coverage_start':'', 'time_type':''}
    # copy METADATA group from ATL11 template. Make lineage/cycle_array conatining each ATL06 file, where the ATL06 filenames
    # with importlib.resources.path('ATL11','package_data') as pp:
    #     template_file=os.path.join(pp, 'atl11_metadata_template.h5')
    template_file = pkg_resources.resource_filename('ATL11','package_data/atl11_metadata_template.h5')
    if os.path.isfile(outfile):
        g = h5py.File(outfile,'r+')
        for ii,infile in enumerate(sorted(infiles)):
            m = h5py.File(template_file,'r')
            if ii==0:
              if 'METADATA' in list(g['/'].keys()):
                  del g['METADATA']
              # get all METADATA groups except Lineage, which we set to zero
              m.copy('METADATA',g)
# fix up Lineage
              if 'Lineage' in list(g['METADATA'].keys()):
                  del g['METADATA']['Lineage']
              g['METADATA'].create_group('Lineage'.encode('ASCII','replace'))
              gf = g['METADATA']['Lineage'].create_group('ANC36-11'.encode('ASCII','replace'))
              gf = g['METADATA']['Lineage'].create_group('ANC38-11'.encode('ASCII','replace'))
              gf = g['METADATA']['Lineage'].create_group('Control'.encode('ASCII','replace'))
# Add in needed root attributes
              create_attribute(gf.id, 'description', [], 'Exact command line execution of ICESat-2/ATL11 algorithm providing all of the conditions required for each individual run of the software.')
              create_attribute(gf.id, 'shortName', [], 'CNTL')
              create_attribute(gf.id, 'version', [], '1')
              create_attribute(gf.id, 'control', [], ' '.join(sys.argv))
# handle METADATA attributes
              create_attribute(g['METADATA/DatasetIdentification'].id, 'fileName', [], os.path.basename(outfile))
              create_attribute(g['METADATA/DatasetIdentification'].id, 'uuid', [], str(uuid.uuid4()))
              create_attribute(g['METADATA/ProcessStep/PGE'].id, 'runTimeParameters', [], ' '.join(sys.argv))
              create_attribute(g['METADATA/ProcessStep/PGE'].id, 'identifier', [], identifier())
              create_attribute(g['METADATA/ProcessStep/PGE'].id, 'softwareDate', [], softwareDate())
              create_attribute(g['METADATA/ProcessStep/PGE'].id, 'softwareTitle', [], softwareTitle())
              gf = g.create_group('quality_assessment'.encode('ASCII','replace'))

              if os.path.isfile(infile):
                f = h5py.File(infile,'r')
                f.copy('quality_assessment/qa_granule_fail_reason',g['quality_assessment'])
                f.copy('quality_assessment/qa_granule_pass_fail',g['quality_assessment'])
# ancillary_data adjustments
                f.copy('ancillary_data',g)
                del g['ancillary_data/land_ice']
                gf = g['METADATA']['Lineage']['Control'].attrs['control'].decode()
                g['ancillary_data/control'][...] = gf.encode('ASCII','replace')
                g['ancillary_data/release'][...] = os.path.basename(outfile).split('_')[3].encode('ASCII','replace')
                g['ancillary_data/version'][...] = os.path.splitext(os.path.basename(outfile))[0].split('_')[4].encode('ASCII','replace')
                
                del g['METADATA/Extent']
                f.copy('METADATA/Extent',g['METADATA'])
                print(sec_offset,start_date)
                if sec_offset is 0:
                  start_delta_time = f['ancillary_data/start_delta_time'][0]
                else:
# To set all date/time to static across ATL11s, plus n seconds
                  epoch_time = datetime(2018,1,1)
                  start_datetime = datetime(start_date[0],start_date[1],start_date[2]) + timedelta(seconds=sec_offset)
                  start_delta_time_object = start_datetime - epoch_time
                  start_delta_time = start_delta_time_object.total_seconds()
                  str_utc = (str(start_datetime.date())+'T'+
                    start_datetime.strftime("%H:%M:%S.%f")+'Z')
                  g['ancillary_data/start_delta_time'][...] = start_delta_time
                  g['ancillary_data/data_start_utc'][...] = str_utc.encode('ASCII','replace')
                  g['ancillary_data/granule_start_utc'][...] = str_utc.encode('ASCII','replace')
                  create_attribute(g['METADATA/Extent'].id, 'rangeBeginningDateTime', [], str_utc)
                create_attribute(g.id, 'short_name', [], 'ATL11')
                for key, keyval in root_info.items():
                       dsname=key
                       if key=='date_created' or key=='history':
                           val=str(datetime.now().date())
                           val=val+'T'+str(datetime.now().time())+'Z'
                           create_attribute(g.id, key, [], val)
                           create_attribute(g['METADATA/ProcessStep/PGE'].id, 'stepDateTime', [], val)
                           create_attribute(g['METADATA/DatasetIdentification'].id, 'creationDate', [], val)
                           continue
                       if key=='identifier_product_format_version':
                           val=softwareVersion()
                           create_attribute(g.id, key, [], val)
                           create_attribute(g['METADATA/ProcessStep/PGE'].id, 'softwareVersion', [], val)
                           create_attribute(g['METADATA/DatasetIdentification'].id, 'VersionID', [], val)
                           create_attribute(g['METADATA/SeriesIdentification'].id, 'VersionID', [], series_version())
                           continue
                       if key=='time_coverage_start':
                           if sec_offset is 0:
                             val = f.attrs[key].decode()
                             create_attribute(g.id, key, [], val)
                           else:
                             create_attribute(g.id, 'time_coverage_start', [], str_utc)
                           continue
                       if key=='granule_type':
                           val = 'ATL11'
                           create_attribute(g.id, key, [], val)
                           continue
                       if key=='level':
                           val = 'L3B'
                           create_attribute(g.id, key, [], val)
                           continue
                       if key=='description':
                           val = f.attrs[key].decode()
                           create_attribute(g.id, key, [], val)
                           continue
                       if key=='time_coverage_end' or key=='time_coverage_duration':
                           continue
                       if dsname in f.attrs:
                           if isinstance(keyval,float):
                             val = f.attrs[key]
                             g.attrs[key]=val
                           else:
                             val = f.attrs[key].decode()
                             create_attribute(g.id, key, [], val)

#
# Read the datasets from orbit_info
                g.create_group('orbit_info'.encode('ASCII','replace'))
#                duplicate_group(f, g, 'orbit_info')
#                g['orbit_info/cycle_number'].dims[0].attach_scale(g['orbit_info/crossing_time'])
#                g['orbit_info/lan'].dims[0].attach_scale(g['orbit_info/crossing_time'])
#                g['orbit_info/orbit_number'].dims[0].attach_scale(g['orbit_info/crossing_time'])
#                g['orbit_info/rgt'].dims[0].attach_scale(g['orbit_info/crossing_time'])
#                g['orbit_info/sc_orient'].dims[0].attach_scale(g['orbit_info/sc_orient_time'])


                m.close()
                f.close()
# Fill orbit_info for each ATL06
#            if ii>0:
#              if os.path.isfile(infile):
#                f = h5py.File(infile,'r')
#                for oi_dset in g['orbit_info'].values():
#                   oi_dset.resize( (oi_dset.shape[0]+1,) )
#                   oi_dset[-1] = f[oi_dset.name][0]
#                f.close()

# Capture ending dates, etc from last ATL06
            if ii==len(infiles)-1:
              if os.path.isfile(infile):
                f = h5py.File(infile,'r')
                for key, keyval in root_info.items():
                    dsname=key
                    if key=='time_coverage_end':
                       val = f.attrs[key].decode()
                       create_attribute(g.id, key, [], val)
                       continue
                    if key=='time_coverage_duration':
                       end_delta_time = f['ancillary_data/end_delta_time'][0]
                       val = float(end_delta_time) - float(start_delta_time)
                       g.attrs[key] = val
                g['ancillary_data/data_end_utc'][...] = f['ancillary_data/data_end_utc']
                g['ancillary_data/end_cycle'][...] = f['ancillary_data/end_cycle']
                g['ancillary_data/end_delta_time'][...] = f['ancillary_data/end_delta_time']
                g['ancillary_data/end_gpssow'][...] = f['ancillary_data/end_gpssow']
                g['ancillary_data/end_gpsweek'][...] = f['ancillary_data/end_gpsweek']
                g['ancillary_data/end_orbit'][...] = f['ancillary_data/end_orbit']
                g['ancillary_data/end_region'][...] = f['ancillary_data/end_region']
                g['ancillary_data/end_rgt'][...] = f['ancillary_data/end_rgt']
                g['ancillary_data/granule_end_utc'][...] = f['ancillary_data/granule_end_utc']
                g['METADATA/Extent'].attrs['rangeEndingDateTime'] = f['METADATA/Extent'].attrs['rangeEndingDateTime']
                  
                m.close()
                f.close()

        g.close()
        poly_buffered_linestring(outfile)
        return()

def poly_buffered_linestring(outfile):
    lonlat_11=[]
    with h5py.File(outfile,'r') as h5f:
        for pair in ['pt1', 'pt2', 'pt3']:
            try:
                lonlat_11 += [np.c_[h5f[pair+'/longitude'], h5f[pair+'/latitude']]]
            except Exception as e:
                print(e)
    print('avg lat lonlat_11[0]',np.sum(lonlat_11[0][:,1])/len(lonlat_11[0]))
    if np.sum(lonlat_11[0][:,1])/len(lonlat_11[0]) >= 0.0:
      polarEPSG=3413
    else:
      polarEPSG=3031

    xformer_ll2pol=pyproj.Transformer.from_crs(4326, polarEPSG)
    xformer_pol2ll=pyproj.Transformer.from_crs(polarEPSG, 4326)
    xy_11=[]
    for ll in lonlat_11:
        xy_11 += [np.c_[xformer_ll2pol.transform(ll[:,1], ll[:,0])]]
    lines=[]
    for xx in xy_11:
        lines += [shapely.geometry.LineString(xx)]
    line_simp=[]
    for line in lines:
        line_simp += [line.simplify(tolerance=100)]
    all_lines=shapely.geometry.MultiLineString(line_simp)
    common_buffer=all_lines.buffer(3000, 4)
    common_buffer=common_buffer.simplify(tolerance=500)

    xpol, ypol = np.array(common_buffer.exterior.coords.xy)
    y1, x1 = xformer_pol2ll.transform(xpol, ypol)
    print("polygon size:",len(x1))

    with h5py.File(outfile,'r+') as h5f:
      if '/orbit_info/bounding_polygon_dim1' in h5f:
        del h5f['/orbit_info/bounding_polygon_dim1']
        del h5f['/orbit_info/bounding_polygon_lon1']
        del h5f['/orbit_info/bounding_polygon_lat1']
      if '/orbit_info/bounding_polygon_dim2' in h5f:
        del h5f['/orbit_info/bounding_polygon_dim2']
        del h5f['/orbit_info/bounding_polygon_lon2']
        del h5f['/orbit_info/bounding_polygon_lat2']

      h5f.create_dataset('/orbit_info/bounding_polygon_dim1',data=np.arange(1,np.size(x1)+1),chunks=True,compression=6,dtype='int32')
      create_attribute(h5f['orbit_info/bounding_polygon_dim1'].id, 'description', [], 'Polygon extent vertex count')
      create_attribute(h5f['orbit_info/bounding_polygon_dim1'].id, 'units', [], '1')
      create_attribute(h5f['orbit_info/bounding_polygon_dim1'].id, 'long_name', [], 'Polygon vertex count')
      create_attribute(h5f['orbit_info/bounding_polygon_dim1'].id, 'source', [], 'model')
      dset = h5f.create_dataset('/orbit_info/bounding_polygon_lon1',data=x1,chunks=True,compression=6,dtype='float32')
      dset.dims[0].attach_scale(h5f['orbit_info']['bounding_polygon_dim1'])
      create_attribute(h5f['orbit_info/bounding_polygon_lon1'].id, 'description', [], 'Polygon extent vertex longitude')
      create_attribute(h5f['orbit_info/bounding_polygon_lon1'].id, 'units', [], 'degrees East')
      create_attribute(h5f['orbit_info/bounding_polygon_lon1'].id, 'long_name', [], 'Polygon vertex longitude')
      create_attribute(h5f['orbit_info/bounding_polygon_lon1'].id, 'source', [], 'model')
      create_attribute(h5f['orbit_info/bounding_polygon_lon1'].id, 'coordinates', [], 'bounding_polygon_dim1')
      dset = h5f.create_dataset('/orbit_info/bounding_polygon_lat1',data=y1,chunks=True,compression=6,dtype='float32')
      dset.dims[0].attach_scale(h5f['orbit_info']['bounding_polygon_dim1'])
      create_attribute(h5f['orbit_info/bounding_polygon_lat1'].id, 'description', [], 'Polygon extent vertex latitude')
      create_attribute(h5f['orbit_info/bounding_polygon_lat1'].id, 'units', [], 'degrees North')
      create_attribute(h5f['orbit_info/bounding_polygon_lat1'].id, 'long_name', [], 'Polygon vertex latitude')
      create_attribute(h5f['orbit_info/bounding_polygon_lat1'].id, 'source', [], 'model')
      create_attribute(h5f['orbit_info/bounding_polygon_lat1'].id, 'coordinates', [], 'bounding_polygon_dim1')

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval
