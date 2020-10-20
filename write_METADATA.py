#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:52:39 2020

@author: root

NOTE: Requires the presence of atl11_metadata_template.h5 in same directory as this python file!
"""
import os, h5py
import numpy as np
import sys
import alphashape
import uuid
from osgeo import osr
from datetime import datetime
#from shapely.ops import cascaded_union
from shapely.ops import unary_union
from shapely.validation import explain_validity
import ATL11
from ATL11.h5util import create_attribute, duplicate_group
from ATL11.version import softwareVersion,softwareDate,softwareTitle,identifier,series_version
from scipy.spatial import ConvexHull

def write_METADATA(outfile,infiles):
    if os.path.isfile(outfile):        
#
# Call filemeta, copies METADATA group from template
#
        filemeta(outfile,infiles)
        g = h5py.File(outfile,'r+')
#        g.create_group('METADATA')
#        gl = g['METADATA'].create_group('Lineage')
#        gf = gl.create_group('ATL06')
        gf = g.create_group('METADATA/Lineage/ATL06'.encode('ASCII','replace'))
        fname = []
        sname = []
        scycle = []
        ecycle = []
        srgt = []
        ergt = []
        sregion = []
        eregion = []
        sgeoseg = []
        egeoseg = []
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
#            version.append(str(digits[2]).encode('ASCII'))
        for pt in g.keys():
            if pt.startswith('pt'):
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
    
def filemeta(outfile,infiles):

    orbit_info={'crossing_time':0., 'cycle_number':0, 'lan':0., \
        'orbit_number':0., 'rgt':0, 'sc_orient':0, 'sc_orient_time':0.}
    root_info={'date_created':'', 'geospatial_lat_max':0., 'geospatial_lat_min':0., \
        'geospatial_lat_units':'', \
        'geospatial_lon_max':0., 'geospatial_lon_min':0., 'geospatial_lon_units':'', \
        'hdfversion':'', 'history':'', \
        'identifier_file_uuid':'', 'identifier_product_format_version':'', 'time_coverage_duration':0., \
        'time_coverage_end':'', 'time_coverage_start':''}
    # copy METADATA group from ATL11 template. Make lineage/cycle_array conatining each ATL06 file, where the ATL06 filenames
    if os.path.isfile(outfile):
        g = h5py.File(outfile,'r+')
        for ii,infile in enumerate(sorted(infiles)):
            m = h5py.File(os.path.dirname(os.path.realpath(__file__))+'/atl11_metadata_template.h5','r')
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
# handle METADATA aatributes
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
                f.copy('ancillary_data',g)
                del g['ancillary_data/land_ice']
                gf = g['METADATA']['Lineage']['Control'].attrs['control'].decode()
                g['ancillary_data/control'][...] = gf.encode('ASCII','replace')
                del g['METADATA/Extent']
                f.copy('METADATA/Extent',g['METADATA'])
                start_delta_time = f['ancillary_data/start_delta_time'][0]
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
                duplicate_group(f, g, 'orbit_info')
                g['orbit_info/cycle_number'].dims[0].attach_scale(g['orbit_info/crossing_time'])
                g['orbit_info/lan'].dims[0].attach_scale(g['orbit_info/crossing_time'])
                g['orbit_info/orbit_number'].dims[0].attach_scale(g['orbit_info/crossing_time'])
                g['orbit_info/rgt'].dims[0].attach_scale(g['orbit_info/crossing_time'])
                g['orbit_info/sc_orient_time'].dims[0].attach_scale(g['orbit_info/sc_orient'])


                m.close()
                f.close()
# Fill orbit_info for each ATL06
            if ii>0:
              if os.path.isfile(infile):
                f = h5py.File(infile,'r')
                for oi_dset in g['orbit_info'].values():
                   oi_dset.resize( (oi_dset.shape[0]+1,) )
                   oi_dset[-1] = f[oi_dset.name][0]
                f.close()

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
        set_polygon_bounds(outfile)
        return()

def set_polygon_bounds(outfile):
    g = h5py.File(outfile,'r+')
    lat = np.empty(0)
    lon = np.empty(0)
    gap_limit = 60.0
    for pt in ['pt1', 'pt2', 'pt3']:
        #if pt doesn't exist, cycle
        if pt not in list(g['/'].keys()):
            continue
        lat = np.concatenate((lat,np.array(g[pt]['latitude']).astype('float')),axis=0)
#        lat = np.concatenate((lat,np.array(g[pt]['crossing_track_data']['latitude']).astype('float')),axis=0)
        lon = np.concatenate((lon,np.array(g[pt]['longitude']).astype('float')),axis=0)
#        lon = np.concatenate((lon,np.array(g[pt]['crossing_track_data']['longitude']).astype('float')),axis=0)

    polar_srs=osr.SpatialReference()
    if np.sum(lat)/len(lat):
      EPSG=3413
    else:
      EPSG=3031
    polar_srs.ImportFromEPSG(EPSG)
    ll_srs=osr.SpatialReference()
    ll_srs.ImportFromEPSG(4326)
    if hasattr(osr,'OAMS_TRADITIONAL_GIS_ORDER'):
      ll_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
      polar_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct=osr.CoordinateTransformation(ll_srs, polar_srs)

    plon, plat, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(lon), np.ravel(lat), np.zeros_like(np.ravel(lat)))]))

    ct=osr.CoordinateTransformation(ll_srs, polar_srs)
    gap_limit = 60.0
    gap_limit_x1, gap_limit_y1, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel([gap_limit]), np.ravel([gap_limit]), np.zeros_like(np.ravel([gap_limit])))]))
    print('gap_limit_x1:',gap_limit_x1)
    gap_limit = gap_limit_x1

    # If max * min is negative, then we span the date line
    if "True" == "True":
# Need to check the lon, lat order is correct!!
# Should be obvious by comparing to plot of groundtrack.
        points1 = tuple(zip(plon, plat))
# Sort by lons, check for large gap
        sort_points = sorted(points1, key=lambda x: x[0])
        sort_lon = np.asarray(sort_points)[:,0]
        sort_lat = np.asarray(sort_points)[:,1]
        print('size sort_lon',np.shape(sort_lon))
        isgap = False
        if len(sort_points) > 10:
          for i in range(3,len(sort_points)-4):
            if abs(sort_lon[i]-sort_lon[i-1]) > gap_limit:
              isgap = True
              index_second = i
              print('lon gap found, index',index_second)
              print(sort_lon[0,],sort_lon[1,])
              break
# If no gap, sort by lats, check for large gap
          if isgap is False:
            for i in range(3,len(sort_points)-4):
              if abs(sort_lat[i]-sort_lat[i-1]) > gap_limit:
                isgap = True
                index_second = i
                print('lat gap found')
                break

# If gap, run alphashpe twice for 2 polygons
        if isgap:
          points1 = tuple(zip(sort_lon[0:index_second-1,],sort_lat[0:index_second-1,]))
          points2 = tuple(zip(sort_lon[index_second::,],sort_lat[index_second::,]))

# Result is a 'alpha shape' Polygon (shapely object)
# , 0. is convex hull
#        alpha_shape = alphashape.alphashape(points1, 0.01)
        alpha_shape1 = alphashape.alphashape(points1, 0.)
#        if isgap: alpha_shape2 = alphashape.alphashape(points2, 0.01)
        if isgap: alpha_shape2 = alphashape.alphashape(points2, 0.)

#print('original size =', len(alpha_shape.exterior.coords))

# But we have way too many points, so we need to simplify. I think the
# 0.0001 is (maybe) in units of degrees. Larger numbers create a smaller
# number of points. I tried different using 10x reductions.
        alpha_shape1 = alpha_shape1.simplify(0.00001)
        alpha_shape1 = alpha_shape1.buffer(0.0001)
        print(explain_validity(alpha_shape1))
        if not alpha_shape1.is_valid:
          print('Fixing invalid 1')
          print(explain_validity(alpha_shape1))
          alpha_shape1 = alpha_shape1.buffer(0)
          if not alpha_shape1.is_valid: print('Still invalid alpha_shape1!')
        if isgap:
          print(explain_validity(alpha_shape2))
          alpha_shape2 = alpha_shape2.simplify(0.00001)
          alpha_shape2 = alpha_shape2.buffer(0.0001)
          if not alpha_shape2.is_valid:
            print('Fixing invalid 2')
            print(explain_validity(alpha_shape2))
            alpha_shape2 = alpha_shape2.buffer(0)

# Extract coordinates as x/y
        x1, y1 = np.array(alpha_shape1.exterior.xy)
        if isgap: x2, y2 = np.array(alpha_shape2.exterior.xy)

# Check for and remove duplicates
    dup_tol = 0.0001
    dup_tol = 90.0 # For polar coords
    del_dups = np.full((len(x1)), False, dtype=bool)
    print("len(x1)",len(x1))
    for i in range(1,len(x1)-1):
      if abs(x1[i]-x1[i-1]) < dup_tol and abs(y1[i]-y1[i-1]) < dup_tol:
        del_dups[i] = True
    x1_nodup = np.delete(x1,np.where(del_dups == 1))
    y1_nodup = np.delete(y1,np.where(del_dups == 1))
    print("len(x1_nodup)",len(x1_nodup))
    x1 = x1_nodup
    y1 = y1_nodup
    if isgap:
      del_dups = np.full((len(x2)), False, dtype=bool)
      print("len(x2)",len(x2))
      for i in range(1,len(x2)-1):
        if abs(x2[i]-x2[i-1]) < dup_tol and abs(y2[i]-y2[i-1]) < dup_tol:
          del_dups[i] = True
      x2_nodup = np.delete(x2,np.where(del_dups == 1))
      y2_nodup = np.delete(y2,np.where(del_dups == 1))
      print("len(x2_nodup)",len(x2_nodup))
      x2 = x2_nodup
      y2 = y2_nodup
      print("len(x2)",len(x2))

    print(explain_validity(alpha_shape1))
    ct=osr.CoordinateTransformation(polar_srs, ll_srs)
    ll_x1, ll_y1, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(x1), np.ravel(y1), np.zeros_like(np.ravel(y1)))]))
    x1 = ll_x1
    y1 = ll_y1
    if isgap:
      ct=osr.CoordinateTransformation(polar_srs, ll_srs)
      ll_x2, ll_y2, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(x2), np.ravel(y2), np.zeros_like(np.ravel(y2)))]))
      x2 = ll_x2
      y2 = ll_y2
# Write polygon info to file
    g.create_dataset('/orbit_info/bounding_polygon_dim1',data=np.arange(1,np.size(x1)+1),chunks=True,compression=6,dtype='int32')
    dset = g.create_dataset('/orbit_info/bounding_polygon_lon1',data=x1,chunks=True,compression=6,dtype='float32')
    dset.dims[0].attach_scale(g['orbit_info']['bounding_polygon_dim1'])
    dset = g.create_dataset('/orbit_info/bounding_polygon_lat1',data=y1,chunks=True,compression=6,dtype='float32')
    dset.dims[0].attach_scale(g['orbit_info']['bounding_polygon_dim1'])
    if isgap:
      g.create_dataset('/orbit_info/bounding_polygon_dim2',data=np.arange(1,np.size(x2)+1),chunks=True,compression=6,dtype='int32')
      dset = g.create_dataset('/orbit_info/bounding_polygon_lon2',data=x2,chunks=True,compression=6,dtype='float32')
      dset.dims[0].attach_scale(g['orbit_info']['bounding_polygon_dim2'])
      dset = g.create_dataset('/orbit_info/bounding_polygon_lat2',data=y2,chunks=True,compression=6,dtype='float32')
      dset.dims[0].attach_scale(g['orbit_info']['bounding_polygon_dim2'])

    g.close()
    return

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
