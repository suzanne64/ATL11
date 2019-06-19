# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py, re, os, csv
import ATL11
from osgeo import osr
import inspect
from PointDatabase import point_data

class group(object):
    # Class to contain an ATL11 structure
    # in ATL11 groups, some datasets have one value per reference point (per_pt_fields)
    # some have one value for each reference point and each cycle (full_fields)
    # some have one value for each point and each polynomial coefficient (poly_fields)
    # all of these are initialized to arrays of the appropriate size, filled with NaNs

    def __init__(self, N_pts, N_cycles, N_coeffs, per_pt_fields=None, full_fields=None, poly_fields=None, xover_fields=None):
        # input variables:
        #  N_pts: Number of reference points to allocate
        #  N_cycles: Number of cycles of data to allocate
        #  N_coeffs: Number of polynomial coefficients to allocate
        #  per_pt_fields: list of fields that have one value per reference point
        #  full_fields: list of fields that have one value per cycle per reference point
        #  poly_fields: list of fields that have one value per polynomial degree combination per reference point

        # assign fields of each type to their appropriate shape and size
        if per_pt_fields is not None:
            for field in per_pt_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, 1]))
        if full_fields is not None:
            for field in full_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, N_cycles]))
        if poly_fields is not None:
            for field in poly_fields:
                setattr(self, field, np.nan + np.zeros([N_pts, N_coeffs]))
        if xover_fields is not None:
            for field in xover_fields:
                setattr(self, field, [])

        # assemble the field names into lists:
        self.per_pt_fields=per_pt_fields
        self.full_fields=full_fields
        self.poly_fields=poly_fields
        self.xover_fields=xover_fields
        self.list_of_fields=self.per_pt_fields+self.full_fields+self.poly_fields+self.xover_fields

    def __repr__(self):
        out=''
        for field in self.list_of_fields:
            out += field+':\n'
            out += str(getattr(self, field))
            out += '\n'
        return out

    def index(self, ind, N_cycles=None, N_coeffs=None, xover_ind=None):
        """
        index an ATL11 data object.
        """
        if N_coeffs is None:
            N_coeffs=self.N_coeffs
        if N_cycles is None:
            N_cycles=self.N_cycles
        try:
            N_pts=len(ind)
        except TypeError:
            N_pts=1
            
        target=ATL11.group(N_pts, N_cycles, N_coeffs, per_pt_fields=self.per_pt_fields.copy(), full_fields=self.full_fields.copy(), poly_fields=self.poly_fields.copy(), xover_fields=self.xover_fields.copy())
                # assign fields of each type to their appropriate shape and size
        if self.per_pt_fields is not None:
            for field in self.per_pt_fields:
                setattr(target, field, getattr(self, field)[ind])
        if self.full_fields is not None:
            for field in self.full_fields:
                setattr(target, field, getattr(self, field)[ind, :])
        if self.poly_fields is not None:
            for field in self.poly_fields:
                setattr(target, field,  getattr(self, field)[ind, :])
        if self.xover_fields is not None:
            # need to pick out the matching crossover-point fields
            for field in self.xover_fields:
                if xover_ind is not None:
                    try:
                        setattr(target, field, getattr(self, field)[xover_ind])
                    except IndexError:
                        setattr(target, field, np.array([]))
                else:
                    setattr(target, field, [])
        return target
class valid_mask:
    # class to hold validity flags for different attributes
    def __init__(self, dims, fields):
        for field in fields:
            setattr(self, field, np.zeros(dims, dtype='bool'))
    def __repr__(self):
        out=''
        for field in dir(self):
            if not field.startswith('__'):
                out += field+':\n'
                temp=getattr(self, field)
                out += str(temp)
                out += '\n'
        return out

class data(object):
    # class to hold ATL11 data in ATL11.groups
    def __init__(self, N_pts=1, N_cycles=1, N_coeffs=9, from_file=None, track_num=None, pair_num=None):
        self.Data=[]
        self.DOPLOT=None

        # define empty records here based on ATL11 ATBD
        # read in parameters information in .csv
        ATL11_root=os.path.dirname(inspect.getfile(ATL11.defaults))
        with open(ATL11_root+'/ATL11_output_attrs.csv','r') as attrfile:
            reader=list(csv.DictReader(attrfile))
        group_names = set([row['group'] for row in reader])
        for group in group_names:
            field_dims=[{k:v for k,v in ii.items()} for ii in reader if ii['group']==group]
            per_pt_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts']
            full_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_cycles']
            poly_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_coeffs']
            xover_fields=[item['field'] for item in field_dims if item['dimensions']=='Nxo']
            setattr(self, group, ATL11.group(N_pts, N_cycles, N_coeffs, per_pt_fields,full_fields,poly_fields, xover_fields))
        self.groups=group_names
        self.slope_change_t0=None
        self.track_num=track_num
        self.pair_num=pair_num
        self.N_cycles=N_cycles
        self.N_coeffs=N_coeffs

    def index(self, ind, N_cycles=None, N_coeffs=None, target=None):
        """
        return a copy of the data for points 'ind'
        """
        try:
            N_pts=len(ind)
        except TypeError:
            N_pts=1
        if N_coeffs is None:
            N_coeffs=self.N_coeffs
        if N_cycles is None:
            N_cycles=self.N_cycles
        if target is None:
            target=ATL11.data(N_pts=N_pts, N_cycles=N_cycles, N_coeffs=N_coeffs, track_num=self.track_num, pair_num=self.pair_num)
        xover_ind=np.in1d(self.crossing_track_data.ref_pt_number, self.corrected_h.ref_pt_number[ind])        
        for group in self.groups:
            setattr(target, group, getattr(self, group).index(ind, N_cycles=N_cycles, N_coeffs=N_coeffs, xover_ind=xover_ind))
        target.poly_exponent=self.poly_exponent.copy()
        return target

    def all_fields(self):
        # return a list of all the fields in an ATL11 instance
        all_vars=[]
        di=vars(self)  # a dictionary
        for item in di.keys():
            if hasattr(getattr(self,item),'list_of_fields'):
                all_vars.append(getattr(getattr(self,item),'list_of_fields'))
        all_vars=[y for x in all_vars for y in x] # flatten list of lists
        return all_vars

    def from_list(self, P11_list):
        # Assemble an ATL11 data instance from a list of ATL11 points.
        # Input: list of ATL11 point instances
        # loop over variables in ATL11.data (self)
        self.__init__(N_pts=len(P11_list), track_num=self.track_num, pair_num=self.pair_num, N_cycles=P11_list[0].corrected_h.cycle_h_shapecorr.shape[1], N_coeffs=P11_list[0].ref_surf.poly_coeffs.shape[1])

        for group in vars(self).keys():
            # check if each variable is an ATl11 group
            if  not isinstance(getattr(self,group), ATL11.group):
                continue
            for field in getattr(self, group).per_pt_fields:
                temp=np.ndarray(shape=[len(P11_list),],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        if 'ref_pt_number' in field:
                            temp[ii]=P11.ref_pt_number
                        else:
                            temp[ii]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)

            for field in getattr(self,group).full_fields:
                temp=np.ndarray(shape=[len(P11_list),P11_list[0].N_cycles],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        temp[ii,:]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)

            for field in getattr(self,group).poly_fields:
                temp=np.ndarray(shape=[len(P11_list),P11_list[0].N_coeffs],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        temp[ii,:]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)
            for field in getattr(self, group).xover_fields:
                temp_out=list()
                for item in P11_list:
                    this_field=getattr(getattr(item, group), field)
                    if len(this_field)>0:
                        temp_out.append(this_field)
                if len(temp_out)>0:
                    try:
                        setattr(getattr(self, group), field, np.concatenate(temp_out).ravel())
                    except ValueError:
                        print("Problem writing %s" %field)

        self.slope_change_t0=P11_list[0].slope_change_t0
        return self

    def from_file(self,  filename, pair=2):
        pt='pt%d' % pair
        with h5py.File(filename,'r') as FH:
            N_pts=FH[pt]['corrected_h']['cycle_h_shapecorr'].shape[0]
            N_cycles=FH[pt]['corrected_h']['cycle_h_shapecorr'].shape[1]
            N_coeffs=FH[pt]['ref_surf']['poly_coeffs'].shape[1]
            self.__init__(N_pts=N_pts, N_cycles=N_cycles, N_coeffs=N_coeffs)
            for group in ('corrected_h','ref_surf','cycle_stats','crossing_track_data'):
                for field in FH[pt][group].keys():
                    try:
                        setattr(getattr(self, group), field, np.array(FH[pt][group][field]))
                    except KeyError:
                        print("ATL11 file %s: missing %s/%s" % (filename, group, field))
            self.poly_exponent={'x':np.array(FH[pt]['ref_surf'].attrs['poly_exponent_x']), 'y':np.array(FH[pt]['ref_surf'].attrs['poly_exponent_y'])}
        return self

    def get_xy(self, proj4_string, EPSG=None):
        # method to get projected coordinates for the data.  Adds 'x' and 'y' fields to the structure
        out_srs=osr.SpatialReference()
        if proj4_string is None and EPSG is not None:
            out_srs.ImportFromEPSG(EPSG)
        else:
            projError= out_srs.ImportFromProj4(proj4_string)
            if projError > 0:
                out_srs.ImportFromWkt(proj4_string)
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        lat=self.corrected_h.ref_pt_lat
        lon=self.corrected_h.ref_pt_lon
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        if lat.size==0:
            self.x=np.zeros_like(lat)
            self.y=np.zeros_like(lon)
        else:
            x, y, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(lon), np.ravel(lat), np.zeros_like(np.ravel(lat)))]))
            self.x=np.reshape(x, lat.shape)
            self.y=np.reshape(y, lon.shape)
        return self

    def write_to_file(self, fileout, params_11=None):
        # Generic code to write data from an ATL11 object to an h5 file
        # Input:
        #   fileout: filename of hdf5 filename to write
        # Optional input:
        #   parms_11: ATL11.defaults structure
        group_name='/pt%d' % self.pair_num
        if os.path.isfile(fileout):
            f = h5py.File(fileout,'r+')
            if group_name in f:
                del f[group_name]
        else:
            f = h5py.File(fileout,'w')
        g=f.create_group(group_name)

        # set the output pair and track attributes
        g.attrs['pair_num']=self.pair_num
        g.attrs['ReferenceGroundTrack']=self.track_num
        g.attrs['N_cycles']=self.N_cycles
        # put default parameters as top level attributes
        if params_11 is None:
            params_11=ATL11.defaults()
        # write each variable in params_11 as an attribute
        for param in  vars(params_11).keys():
            try:
                g.attrs[param]=getattr(params_11, param)
            except:
                print("write_to_file:could not automatically set parameter: %s" % param)

        # put groups, fields and associated attributes from .csv file
        with open(os.path.dirname(inspect.getfile(ATL11.data))+'/ATL11_output_attrs.csv','r') as attrfile:
            reader=list(csv.DictReader(attrfile))
        group_names=set([row['group'] for row in reader])
        attr_names=[x for x in reader[0].keys() if x != 'field' and x != 'group']
        field_attrs = {row['field']: {attr_names[ii]:row[attr_names[ii]] for ii in range(len(attr_names))} for row in reader}
        for group in group_names:
            if hasattr(getattr(self,group),'list_of_fields'):
                grp = g.create_group(group)
                if 'ref_surf' in group:
                    grp.attrs['poly_exponent_x']=np.array([item[0] for item in params_11.poly_exponent_list], dtype=int)
                    grp.attrs['poly_exponent_y']=np.array([item[1] for item in params_11.poly_exponent_list], dtype=int)
                    grp.attrs['slope_change_t0'] =np.mean(self.slope_change_t0).astype('int')
                    g.attrs['N_poly_coeffs']=int(self.N_coeffs)
                list_vars=getattr(self,group).list_of_fields
                if list_vars is not None:
                    for field in list_vars:
                        dset = grp.create_dataset(field,data=getattr(getattr(self,group),field))
                        for attr in attr_names:
                            dset.attrs[attr] = field_attrs[field][attr]
        f.close()
        return


    def get_xovers(self, rgt, cycle, pair):
        xo={'ref':{},'crossing':{},'both':{}}
        for field in ['time','h','h_sigma','ref_pt_number','rgt','PT','atl06_quality_summary','latitude','longitude']:
            xo['ref'][field]=[]
            xo['crossing']['field']=[]
    
        for i1, ref_pt in enumerate(self.crossing_track_data.ref_pt_number):
            i0=np.where(self.corrected_h.ref_pt_number==ref_pt)[0][0]
            for ic in range(self.corrected_h.mean_cycle_time.shape[1]):
                if not np.isfinite(self.corrected_h.cycle_h_shapecorr[i0, ic]):
                    continue
                xo['ref']['latitude'] += [self.corrected_h.ref_pt_lat[i0]]
                xo['ref']['longitude'] += [self.corrected_h.ref_pt_lon[i0]]
                
                xo['ref']['time'] += [self.corrected_h.mean_cycle_time[i0, ic]]
                xo['ref']['h']    += [self.corrected_h.cycle_h_shapecorr[i0, ic]]
                xo['ref']['h_sigma']    += [self.corrected_h.cycle_h_shapecorr_sigma[i0, ic]]
                xo['ref']['ref_pt_number'] += [self.corrected_h.ref_pt_number[i0]]
                xo['ref']['rgt'] += [rgt]
                xo['ref']['PT'] += [pair]  
                xo['ref']['atl06_quality_summary'] += [self.cycle_stats.atl06_quality_summary_zero_count[i0, ic] >1]
                
                xo['crossing']['time'] += [self.crossing_track_data.delta_time[i1]]
                xo['crossing']['h']  +=  [self.crossing_track_data.h_shapecorr[i1]]
                xo['crossing']['h_sigma']  +=  [self.crossing_track_data.h_shapecorr_sigma[i1]]
                xo['crossing']['ref_pt_number'] += [self.crossing_track_data.ref_pt_number[i1]]
                xo['crossing']['rgt'] += [self.crossing_track_data.rgt_crossing[i1]]
                xo['crossing']['PT'] += [self.crossing_track_data.pt_crossing[i1]]
                xo['crossing']['atl06_quality_summary'] += [self.crossing_track_data.atl06_quality_summary[i1]]
                xo['crossing']['RSSz']  += [self.crossing_track_data.along_track_diff_rss[i1]]
        xo['crossing']['latitude']=xo['ref']['latitude']
        xo['crossing']['longitude']=xo['ref']['longitude']
        return point_data().from_dict(xo['ref']), point_data().from_dict(xo['crossing']) 



    def plot(self):
        # method to plot the results.  At present, this plots corrected h AFN of x_atc
        n_cycles=self.corrected_h.cycle_h_shapecorr.shape[1]
        HR=np.nan+np.zeros((n_cycles, 2))
        h=list()
        #plt.figure(1);plt.clf()
        for cycle in range(n_cycles):
            xx=self.ref_surf.ref_pt_x_atc
            zz=self.corrected_h.cycle_h_shapecorr[:,cycle]
            ss=self.corrected_h.cycle_h_shapecorr_sigma[:,cycle]
            good=np.abs(ss)<15
            ss[~good]=np.NaN
            zz[~good]=np.NaN
            if np.any(good):
                h0=plt.errorbar(xx[good],zz[good],ss[good], marker='o',picker=5)
                h.append(h0)
                HR[cycle,:]=np.array([zz[good].min(), zz[good].max()])
                #plt.plot(xx[good], zz[good], 'k',picker=None)
        temp=self.corrected_h.cycle_h_shapecorr.copy()
        temp[self.corrected_h.cycle_h_shapecorr_sigma>20]=np.nan
        temp=np.nanmean(temp, axis=1)
        plt.plot(xx, temp, 'k.', picker=5)
        plt.ylim((np.nanmin(HR[:,0]),  np.nanmax(HR[:,1])))
        plt.xlim((np.nanmin(xx),  np.nanmax(xx)))
        return h
