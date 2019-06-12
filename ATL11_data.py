# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py, re, os, csv
from ATL11 import ATL11_defaults
from osgeo import osr
import inspect

class ATL11_group(object):
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

    def index(self, pts):
        target=ATL11_data(1, self.N_cycles, self.N_coeffs, per_pt_fields=self.per_pt_fields.copy(), full_fields=self.full_fields.copy(), poly_fields=self.poly_fields.copy(), xover_fields=self.xover_fields.copy())
                # assign fields of each type to their appropriate shape and size
        if self.per_pt_fields is not None:
            for field in self.per_pt_fields:
                setattr(target, field, np.nan + np.zeros([self.N_pts, 1]))
        if self.full_fields is not None:
            for field in self.full_fields:
                setattr(target, field, np.nan + np.zeros([self.N_pts, self.N_cycles]))
        if self.poly_fields is not None:
            for field in self.poly_fields:
                setattr(target, field, np.nan + np.zeros([self.N_pts, self.N_coeffs]))
        if self.xover_fields is not None:
            for field in self.xover_fields:
                setattr(self, field, [])
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

class ATL11_data(object):
    # class to hold ATL11 data in ATL11_groups
    def __init__(self, N_pts=1, N_cycles=1, N_coeffs=9, from_file=None, track_num=None, pair_num=None):
        self.Data=[]
        self.DOPLOT=None

        # define empty records here based on ATL11 ATBD
        # read in parameters information in .csv
        ATL11_root=os.path.dirname(inspect.getfile(ATL11_defaults))
        with open(ATL11_root+'/ATL11_output_attrs.csv','r') as attrfile:
            reader=list(csv.DictReader(attrfile))
        group_names = set([row['group'] for row in reader])
        for group in group_names:
            field_dims=[{k:v for k,v in ii.items()} for ii in reader if ii['group']==group]
            per_pt_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts']
            full_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_cycles']
            poly_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_coeffs']
            xover_fields=[item['field'] for item in field_dims if item['dimensions']=='Nxo']
            setattr(self, group, ATL11_group(N_pts, N_cycles, N_coeffs, per_pt_fields,full_fields,poly_fields, xover_fields))
        self.groups=group_names
        self.slope_change_t0=None
        self.track_num=track_num
        self.pair_num=pair_num
        self.N_cycles=N_cycles
        self.N_coeffs=N_coeffs

    def index(self, ind):
        """
        return a copy of the data for points 'ind'
        """
        try:
            N_pts=len(ind)
        except TypeError:
            N_pts=1
        target=ATL11_data(N_pts=N_pts, N_cycles=self.N_cycles, N_coeffs=self.N_coeffs, track_num=self.track_num, pair_num=self.pair_num)
        for group in self.groups:
            setattr(target, group, getattr(self, group).index(ind))
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
        # loop over variables in ATL11_data (self)
        self.__init__(N_pts=len(P11_list), track_num=self.track_num, pair_num=self.pair_num, N_cycles=P11_list[0].corrected_h.cycle_h_shapecorr.shape[1], N_coeffs=P11_list[0].ref_surf.poly_coeffs.shape[1])

        for group in vars(self).keys():
            # check if each variable is an ATl11 group
            if  not isinstance(getattr(self,group), ATL11_group):
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

    def from_file(self, pair, filename=None):
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
        #   parms_11: ATL11_defaults structure
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

        # put default parameters as top level attributes
        if params_11 is None:
            params_11=ATL11_defaults()
        # write each variable in params_11 as an attribute
        for param in  vars(params_11).keys():
            try:
                g.attrs[param]=getattr(params_11, param)
            except:
                print("write_to_file:could not automatically set parameter: %s" % param)

        # put groups, fields and associated attributes from .csv file
        with open(os.path.dirname(inspect.getfile(ATL11_data))+'/ATL11_output_attrs.csv','r') as attrfile:
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
                list_vars=getattr(self,group).list_of_fields
                if list_vars is not None:
                    for field in list_vars:
                        dset = grp.create_dataset(field,data=getattr(getattr(self,group),field))
                        for attr in attr_names:
                            dset.attrs[attr] = field_attrs[field][attr]
        f.close()
        return

#    def get_xovers(self):




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
