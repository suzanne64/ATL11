
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:33 2017f

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py,  os, csv
import ATL11
from osgeo import osr
import pointCollection as pc
from ATL11.ATL06_pair import ATL06_pair
from ATL11.h5util import create_attribute
from importlib import resources


class data(object):
    # class to hold ATL11 data in ATL11.groups
    def __init__(self, N_pts=1, cycles=[1,2], N_coeffs=9, from_file=None, track_num=None, beam_pair=None):
        self.Data=[]
        self.DOPLOT=None

        # define empty records here based on ATL11 ATBD
        # read in parameters information in .csv
        attrfile = str(resources.files('ATL11').joinpath("package_data/ATL11_output_attrs.csv") )
        # csv.DictReader should be wrapped in a list to iterate
        reader=list(csv.DictReader(open(attrfile)))

        group_names = set([row['group'] for row in reader])
        for group in group_names:
            field_dims=[{k:v for k,v in ii.items()} for ii in reader if ii['group']==group]
            per_pt_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts']
            full_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_cycles']
            poly_fields=[item['field'] for item in field_dims if item['dimensions']=='N_pts, N_coeffs']
            xover_fields=[item['field'] for item in field_dims if item['dimensions']=='Nxo']
            setattr(self, group, ATL11.group(N_pts, cycles, N_coeffs, per_pt_fields,full_fields,poly_fields, xover_fields))

        self.groups=group_names
        self.track_num=track_num
        self.beam_pair=beam_pair
        self.pair_num=beam_pair
        self.cycles=cycles
        if cycles is not None:
            self.cycle_number=np.arange(cycles[0], cycles[1]+1)
        else:
            self.cycle_number=None
        self.N_coeffs=N_coeffs
        self.Nxo=0
        self.attrs={}
        self.filename=None

    def assign(self, var_dict):
        for key in var_dict:
            setattr(self, key, var_dict[key])

    def index(self, ind, cycles=None, N_coeffs=None, target=None):
        """
        return a copy of the data for points 'ind'
        """
        try:
            N_pts=len(ind)
        except TypeError:
            N_pts=1
        if N_coeffs is None:
            N_coeffs=self.N_coeffs
        if cycles is None:
            cycles=self.cycles
        if target is None:
            target=ATL11.data(N_pts=N_pts, cycles=cycles, N_coeffs=N_coeffs, track_num=self.track_num, beam_pair=self.beam_pair)
        xover_ind=np.in1d(self.crossing_track_data.ref_pt, self.ROOT.ref_pt[ind])
        for group in self.groups:
            setattr(target, group, getattr(self, group).index(ind, cycles=cycles, N_coeffs=N_coeffs, xover_ind=xover_ind))
        target.poly_exponent=self.poly_exponent.copy()
        if hasattr(self,'x'):
            setattr(target,'x', self.x[ind])
            setattr(target,'y', self.y[ind])
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
        self.__init__(N_pts=len(P11_list), track_num=self.track_num, beam_pair=self.beam_pair, cycles=P11_list[0].cycles, N_coeffs=P11_list[0].ref_surf.poly_coeffs.shape[1])

        for group in vars(self).keys():
            # check if each variable is an ATl11 group
            if  not isinstance(getattr(self,group), ATL11.group):
                continue
            for field in getattr(self, group).per_pt_fields:
                temp=np.ndarray(shape=[len(P11_list),],dtype=float)
                for ii, P11 in enumerate(P11_list):
                    if hasattr(getattr(P11,group),field):
                        if 'ref_pt' in field:
                            temp[ii]=P11.ref_pt
                        else:
                            temp[ii]=getattr(getattr(P11,group), field)
                setattr(getattr(self,group),field,temp)

            for field in getattr(self,group).full_fields:
                temp=np.ndarray(shape=[len(P11_list),P11_list[0].cycles[1]-P11_list[0].cycles[0]+1],dtype=float)
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

        return self

    def from_list_of_ATL11_data(self, D11_list):
        # Assemble an ATL11 data instance from a list of ATL11 data
        # Input: list of ATL11 data instances
        # loop over variables in ATL11.data (self), concatenate each with the
        # proper dimensions

        # get the point count
        ref_pts=np.concatenate([Di.ROOT.ref_pt for Di in D11_list])
        N_pts=len(ref_pts)
        self.__init__(N_pts=N_pts, track_num=self.track_num,
                      beam_pair=self.beam_pair,
                      cycles=D11_list[0].cycles,
                      N_coeffs=D11_list[0].ref_surf.poly_coeffs.shape[1])

        for group in vars(self).keys():
            # check if each variable is an ATl11 group
            if  not isinstance(getattr(self,group), ATL11.group):
                continue
            for field in getattr(self, group).per_pt_fields:
                temp=np.concatenate([
                    getattr(getattr(Di, group), field) for Di in D11_list])
                setattr(getattr(self,group),field,temp)

            for field in getattr(self,group).full_fields:
                temp=np.concatenate([
                    getattr(getattr(Di, group), field) for Di in D11_list], axis=0)
                setattr(getattr(self,group),field,temp)

            for field in getattr(self,group).poly_fields:
                temp=np.concatenate([
                    getattr(getattr(Di, group), field) for Di in D11_list], axis=0)
                setattr(getattr(self,group),field,temp)

            for field in getattr(self, group).xover_fields:
                temp_out=list()
                for item in D11_list:
                    this_field=getattr(getattr(item, group), field)
                    if len(this_field)>0:
                        temp_out.append(this_field)
                if len(temp_out)>0:
                    try:
                        setattr(getattr(self, group), field, np.concatenate(temp_out).ravel())
                    except ValueError:
                        print("Problem writing %s" %field)

        return self


    def from_file(self,  filename, pair=2, index_range=[0, -1], field_dict=None, invalid_to_nan=True):
        '''
        read ATL11 data for a pair track from a file
        '''
        self.filename=filename
        self.no_pair = 0
        #index_range=slice(index_range[0], index_range[1]);
        pt='pt%d' % pair
        with h5py.File(filename,'r') as FH:
            if pt not in FH:
                self.no_pair = 1
                return self
            # if the field dict is not specified, read it
            if field_dict is None:
                field_dict={}
                field_dict['ROOT']=[]
                for key,val in FH[pt].items():
                    if isinstance(val, h5py.Group):
                        field_dict[key]=[]
                        for field in FH[pt][key].keys():
                            field_dict[key].append(field)
                    if isinstance(val, h5py.Dataset):
                        field_dict['ROOT'].append(key)

            N_pts=FH[pt]['h_corr'][index_range[0]:index_range[-1],:].shape[0]
            cycles=[FH[pt].attrs['first_cycle'], FH[pt].attrs['last_cycle']]
            N_coeffs=FH[pt]['ref_surf']['poly_coeffs'].shape[1]
            self.__init__(N_pts=N_pts, cycles=cycles, N_coeffs=N_coeffs)

            for group in (field_dict):
#                if in_group is None:
#                    out_group=in_group
#                else:
#                    out_group='ROOT'
                if group != 'crossing_track_data':
                    if group == 'ROOT':
                        for field in field_dict[group]:
                            try:
                                this_field = np.array(FH[pt][field]).astype('float')
                                # check for invalids replace with nans
                                if invalid_to_nan:
                                    this_field[this_field==FH[pt][field].fillvalue.astype('float')] = np.nan
                                if len(this_field.shape) > 1:
                                    setattr(getattr(self, group), field, this_field[index_range[0]:index_range[1],:])
                                else:
                                    if 'cycle_number' in field: # and (out_group=='ROOT' or out_group=='cycle_stats'):
                                        setattr(getattr(self, group), field, this_field[:])
                                    else:
                                        setattr(getattr(self, group), field, this_field[index_range[0]:index_range[1]])
                            except KeyError:
                                print("ATL11 file %s: missing %s/%s" % (filename, group, field))

                    else:
                        for field in field_dict[group]:
                            try:
                                this_field = np.array(FH[pt][group][field]).astype('float')
                                # check for invalids replace with nans
                                if invalid_to_nan:
                                    this_field[this_field==FH[pt][group][field].fillvalue.astype('float')] = np.nan
                                if len(this_field.shape) > 1:
                                    setattr(getattr(self, group), field, this_field[index_range[0]:index_range[1],:])
                                else:
                                    setattr(getattr(self, group), field, this_field[index_range[0]:index_range[1]])
                            except KeyError:
                                print("ATL11 file %s: missing %s/%s" % (filename, group, field))
                else:
                    # get the indices for the crossing_track_data group:
                    if self.ROOT.ref_pt.size <1:
                        continue
                    xing_ref_pt = np.array(FH[pt]['crossing_track_data']['ref_pt'])
                    xing_ind = np.flatnonzero( (xing_ref_pt >= self.ROOT.ref_pt[0]) & \
                                      (xing_ref_pt <= self.ROOT.ref_pt[-1]) )
                    for field in field_dict['crossing_track_data']:
                        try:
                            this_field = np.array(FH[pt][group][field]).astype('float')
                            # check for invalids replace with nans
                            if invalid_to_nan:
                                this_field[this_field==FH[pt][group][field].fillvalue.astype('float')] = np.nan

                            setattr(getattr(self, group), field, \
                                    np.array(this_field[list(xing_ind)]))
                                    #np.array(FH[pt]['crossing_track_data'][field][list(xing_ind)]))
                        except KeyError:
                            print("ATL11 file %s: missing %s/%s" % (filename, 'crossing_track_data', field))
                        except ValueError:
                            print("ATL11 file %s: misshapen %s/%s" % (filename, 'crossing_track_data', field))

            self.poly_exponent={'x':np.array(FH[pt]['ref_surf']['poly_exponent_x']), 'y':np.array(FH[pt]['ref_surf']['poly_exponent_y'])}
            for attr in FH[pt].attrs.keys():
                self.attrs[attr]=FH[pt].attrs[attr]
            self.cycle_number=np.array(FH[pt]['cycle_number'])
        return self

    def get_xy(self, proj4_string=None, EPSG=None):
        lat=self.ROOT.latitude
        lon=self.ROOT.longitude
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
        if hasattr(osr,'OAMS_TRADITIONAL_GIS_ORDER'):
            ll_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        if lat.size==0:
            self.x=np.zeros_like(lat)
            self.y=np.zeros_like(lon)
        else:
            x, y, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(lon), np.ravel(lat), np.zeros_like(np.ravel(lat)))]))
            self.x=np.reshape(x, lat.shape)
            self.y=np.reshape(y, lon.shape)
            bad=~np.isfinite(x)
            self.x[bad]=np.nan
            self.y[bad]=np.nan
        return self

    def write_to_file(self, fileout, params_11=None, write_xovers=False, cycle=None):
        # Generic code to write data from an ATL11 object to an h5 file
        # Input:
        #   fileout: filename of hdf5 filename to write
        # Optional input:
        #   parms_11: ATL11.defaults structure

        if cycle is None:
            cycles = self.cycles
        else:
            cycles = [cycle, cycle]

        beam_pair_name='/pt%d' % self.beam_pair
        beam_pair_name=beam_pair_name.encode('ASCII')

        # read groups, fields and associated attributes from .csv file

        attrfile=os.path.join(str(resources.files('ATL11')), 'package_data', 'ATL11_output_attrs.csv')
        with open(attrfile, 'r') as attr_fh:
            reader=list(csv.DictReader(attr_fh))
        group_names=set([row['group'] for row in reader])
        attr_names=[x for x in reader[0].keys() if x != 'field' and x != 'group']
        group_descriptions = {}
        for group in group_names:
            group_descriptions[group] = [jj['description'] for jj in reader if jj['group']==group and jj['field']=='']

        if os.path.isfile(fileout):
            f = h5py.File(fileout.encode('ASCII'),'r+')
            if beam_pair_name in f:
                del f[beam_pair_name]
        else:
            f = h5py.File(fileout.encode('ASCII'),'w')
        g=f.create_group(beam_pair_name)

        # set the output pair and track attributes
        g.attrs['beam_pair'.encode('ASCII')]=self.beam_pair
        g.attrs['ReferenceGroundTrack'.encode('ASCII')]=self.track_num
        g.attrs['first_cycle'.encode('ASCII')]=self.cycles[0]
        g.attrs['last_cycle'.encode('ASCII')]=self.cycles[1]
        g.attrs['description'.encode('ASCII')] = \
                group_descriptions['ROOT'][0].replace('%beam_pair%', str(self.beam_pair)).encode()
        # put default parameters as top level attributes
        if params_11 is None:
            params_11=ATL11.defaults()

        # write each variable in params_11 as an attribute
        for param, val in  vars(params_11).items():
            if param=="poly_exponent_list":
                continue
            if not isinstance(val,(dict,type(None))):
                try:
                    if param == 'ATL06_xover_field_list' and write_xovers:
                        xover_attr=getattr(params_11, param)
                        xover_attr = [x.encode('ASCII') for x in xover_attr]
                        g.attrs[param.encode('ASCII')]=xover_attr
                    else:
                        g.attrs[param.encode('ASCII')]=getattr(params_11, param)
                except Exception as e:
                    print("write_to_file:could not automatically set parameter: %s error = %s" % (param,str(e)))
                    continue


        # start with 'ROOT' group
        list_vars=getattr(self,'ROOT').list_of_fields
        list_vars.append('cycle_number')

        def field_attrs_for_group(group, reader, attr_names):
            field_attrs = {row['field']: {attr_names[ii]:row[attr_names[ii]] for ii in range(len(attr_names))}
                           for row in reader if group in row['group'] and len(row['field']) > 0}
            for field in field_attrs:
                dtype = field_attrs[field]['datatype'].lower()
                if 'int' in dtype:
                    fill_val = np.iinfo(np.dtype(dtype)).max
                else:
                    fill_val = np.finfo(np.dtype(dtype)).max
                field_attrs[field]['fill_val'] = fill_val
                field_attrs[field]['dtype'] = dtype
            return field_attrs

        # establish the two main dimension scales

        field_attrs = field_attrs_for_group('ROOT', reader, attr_names)
        for field in ['ref_pt','cycle_number']:
            dimensions = field_attrs[field]['dimensions'].split(',')
            data = getattr(getattr(self,'ROOT'),field)
            dset = g.create_dataset(field.encode('ASCII'),data=data,chunks=True,compression=6,dtype=field_attrs[field]['dtype']) #,fillvalue=fillvalue)
            dset.dims[0].label = field

            for attr in attr_names:
                if 'dimensions' not in attr and 'datatype' not in attr:
                    if 'valid_m' in attr:
                        cast_to = np.dtype(field_attrs[field]['dtype'])
                        if not field_attrs[field][attr]=='':
                                dset.attrs[attr] = cast_to.type(field_attrs[field][attr])
                    else:
                        create_attribute(dset.id, attr, [], str(field_attrs[field][attr]))
            dset.attrs['_FillValue'.encode('ASCII')] = field_attrs[field]['fill_val']

        for field in [item for item in list_vars if (item != 'ref_pt') and (item != 'cycle_number')]:
            dimensions = field_attrs[field]['dimensions'].split(',')
            data = getattr(getattr(self,'ROOT'),field)
            # change nans to proper invalid, depending on datatype
            data = np.nan_to_num(data,nan=field_attrs[field]['fill_val'])
            if field_attrs[field]['datatype'].startswith('int'):
                data = data.astype('int')  # don't change to int before substituting nans with invalid.
            dset = g.create_dataset(field.encode('ASCII'),
                                    data=data,fillvalue=field_attrs[field]['fill_val'],
                                    chunks=True,
                                    compression=6,
                                    dtype=field_attrs[field]['dtype'])
            dset.dims[0].label = field

            for ii,dim in enumerate(dimensions):
                dim=dim.strip()
                if 'N_pts' in dim:
                    dset.dims[ii].attach_scale(g['ref_pt'])
                    dset.dims[ii].label = 'ref_pt'
                if 'N_cycles' in dim:
                    dset.dims[ii].attach_scale(g['cycle_number'])
                    dset.dims[ii].label = 'cycle_number'
            for attr in attr_names:
                if 'dimensions' not in attr and 'datatype' not in attr:
                    if 'valid_m' in attr:
                        cast_to = np.dtype(field_attrs[field]['dtype'])
                        if not field_attrs[field][attr]=='':
                                dset.attrs[attr] = cast_to.type(field_attrs[field][attr])
                    else:
                        create_attribute(dset.id, attr, [], str(field_attrs[field][attr]))
            dset.attrs['_FillValue'.encode('ASCII')] = field_attrs[field]['fill_val']

        for group in [item for item in group_names if item != 'ROOT']:
            # skip crossing_track_data unless write_xovers is specified
            if group == "crossing_track_data" and not write_xovers:
                continue

            field_attrs = field_attrs_for_group(group, reader, attr_names)
            if hasattr(getattr(self,group),'list_of_fields'):
                grp = g.create_group(group.encode('ASCII'))
                if len(group_descriptions[group]) > 0:
                    grp.attrs['description'.encode('ASCII')]=group_descriptions[group][0].encode()

                if 'crossing_track_data' in group:
                    this_ref_pt=getattr(getattr(self,group),'ref_pt')
                    if len(this_ref_pt) > 0:
                        dset = grp.create_dataset('ref_pt'.encode('ASCII'),data=this_ref_pt,chunks=True,compression=6,dtype=field_attrs['ref_pt']['datatype'])
                    else:
                        dset = grp.create_dataset('ref_pt'.encode('ASCII'), shape=[0],chunks=True,compression=6,dtype=np.int32)
                    dset.dims[0].label = 'ref_pt'.encode('ASCII')
                    for attr in attr_names:
                        if 'dimensions' not in attr and 'datatype' not in attr:
                            create_attribute(dset.id, attr, [], field_attrs['ref_pt'][attr])

                if 'ref_surf' in group:
                    dset = grp.create_dataset('poly_exponent_x'.encode('ASCII'),data=np.array([item[0] for item in params_11.poly_exponent_list]),chunks=True,compression=6,dtype=field_attrs['poly_exponent_x']['datatype'])
                    dset.dims[0].label = 'poly_exponent_x'.encode('ASCII')
                    dset = grp.create_dataset('poly_exponent_y'.encode('ASCII'),data=np.array([item[1] for item in params_11.poly_exponent_list]),chunks=True,compression=6, dtype=field_attrs['poly_exponent_y']['datatype'])
                    dset.dims[0].label = 'poly_exponent_y'.encode('ASCII')
                    for field in ['poly_exponent_x', 'poly_exponent_y']:
                        dset=grp[field]
                        for attr in attr_names:
                            if 'dimensions' not in attr and 'datatype' not in attr:
                                if 'valid_m' in attr:
                                    cast_to = np.dtype(field_attrs[field]['dtype'])
                                    if not field_attrs[field][attr]=='':
                                            dset.attrs[attr] = cast_to.type(field_attrs[field][attr])
                                else:
                                    create_attribute(dset.id, attr, [], str(field_attrs[field][attr]))

                    grp.attrs['poly_exponent_x'.encode('ASCII')]=np.array([item[0] for item in params_11.poly_exponent_list], dtype=int)
                    grp.attrs['poly_exponent_y'.encode('ASCII')]=np.array([item[1] for item in params_11.poly_exponent_list], dtype=int)
                    g.attrs['N_poly_coeffs'.encode('ASCII')]=int(self.N_coeffs)

                list_vars=getattr(self,group).list_of_fields
                if group == 'crossing_track_data':
                    list_vars.remove('ref_pt')  # handled above
                if list_vars is not None:
                    for field in list_vars:
                        dimensions = field_attrs[field]['dimensions'].split(',')
                        data = getattr(getattr(self,group),field)
                        data = np.nan_to_num(data,nan=field_attrs[field]['fill_val'])
                        # change nans to proper invalid, depending on datatype
                        if field_attrs[field]['datatype'].startswith('int'):
                            data = data.astype('int')  # don't change to int before substituting nans with invalid.

                        dset = grp.create_dataset(field.encode('ASCII'),data=data,
                                                  fillvalue=field_attrs[field]['fill_val'],
                                                  chunks=True,
                                                  compression=6,
                                                  dtype=field_attrs[field]['dtype'])
                        for ii,dim in enumerate(dimensions):
                            dim=dim.strip()
                            if 'N_pts' in dim:
                                dset.dims[ii].attach_scale(g['ref_pt'])
                                dset.dims[ii].label = 'ref_pt'
                            if 'N_cycles' in dim:
                                dset.dims[ii].attach_scale(g['cycle_number'])
                                dset.dims[ii].label = 'cycle_number'
                            if 'N_coeffs' in dim:
                                dset.dims[ii].attach_scale(grp['poly_exponent_x'])
                                dset.dims[ii].attach_scale(grp['poly_exponent_y'])
                                dset.dims[ii].label = '(poly_exponent_x, poly_exponent_y)'
                            if 'Nxo' in dim:
                                dset.dims[ii].attach_scale(grp['ref_pt'])
                                dset.dims[ii].label = 'ref_pt'
                        for attr in attr_names:
                            if 'dimensions' not in attr and 'datatype' not in attr:
                                if 'valid_m' in attr:
                                    cast_to = np.dtype(field_attrs[field]['dtype'])
                                    if not field_attrs[field][attr]=='':
                                        dset.attrs[attr] = cast_to.type(field_attrs[field][attr])
                                else:
                                    create_attribute(dset.id, attr, [], str(field_attrs[field][attr]))

                        dset.attrs['_FillValue'.encode('ASCII')] = field_attrs[field]['fill_val']

        f.close()
        return

    def write_xover_files(self, args, replace=[False]):
        '''
        Write crossing track data to temporary hdf5 files.

        Parameters
        ----------
        args : dataspace
            arguments defined by argparse.
        replace : list, optional
            List containing one boolean that describes whether the output files
            should be replaced. The default is [False].

        Returns
        -------
        None.

        '''
        ref, crossing = self.get_xovers()
        if crossing is None or crossing.cycle_number.size==0:
            return
        _, d_cyc = pc.unique_by_rows(crossing.cycle_number,
                                     return_dict=True)
        ref_context_fields={'ref_pt','cycle_number','rgt', 'latitude','longitude',
                        'pair_track', 'fit_quality','dem_h','geoid_h',
                        'n_slope','e_slope','x_atc','y_atc'}
        for cycle, ind in d_cyc.items():
            cycle_dir = os.path.join(args.xover_output_dir,f'cycle_{int(cycle[0]):02d}')
            try:
                os.mkdir(cycle_dir)
            except FileExistsError:
                pass
            this_out_file="%s/ATL11_atxo_%04d%02d_%02d_%03d_%02d.h5" %( \
                cycle_dir, args.rgt, args.subproduct, int(cycle[0]), \
                args.Release, args.Version)
            if args.verbose:
                print(f"writing xovers to {this_out_file}")
            # sort index by reference point
            ind=ind[np.argsort(ref.ref_pt[ind,0])]
            # write crossing track data
            temp=crossing[ind]
            # rename beam_pair to pair_track
            temp.assign(pair_track=temp.beam_pair.copy())
            temp.fields.remove('beam_pair')
            #N.B.  ref_pair is not an output variable.  It is there as an internal check
            temp.assign(ref_pair = np.zeros(temp.shape, dtype=int)+int(self.pair_num))
            temp.assign(ref_rgt = np.zeros(temp.shape, dtype=int)+int(self.track_num))
            temp.ravel_fields()
            temp.to_h5(this_out_file, group=f'pt{self.pair_num}/crossing_track', replace=replace[0])

            col = np.flatnonzero(self.cycle_number==cycle)
            if len(col)==0:
                # if a cycle is not present in the input cycles, fill ref with NaNs (except cycle and ref_pt fields)
                col=0
                fill_nans = True
            else:
                fill_nans = False

            # subset the reference track
            try:
                temp=ref[ind,:][:,col]
            except Exception as e:
                print(e)
            # rename beam_pair to pair_track
            temp.assign(pair_track=temp.beam_pair.copy())
            temp.fields.remove('beam_pair')
            if fill_nans:
                temp.cycle_number[:]=cycle
                for field in temp.fields:
                    temp_f = getattr(temp, field)
                    if field not in ref_context_fields:
                        temp_f = temp_f.astype(float)+np.nan
                    setattr(temp, field, temp_f)

            temp.ravel_fields()
            temp.to_h5(this_out_file, group=f'pt{self.pair_num}/datum_track', replace=False)
        replace[0]=False

    def get_xovers(self,invalid_to_nan=True, calc_delta=False):
        '''


        Parameters
        ----------
        invalid_to_nan : bool, optional
            If true, invalid values will be converted to NaNs. The default is True.
        calc_delta : bool, optional
            If True, the differenecs between reference and crossing-track values will be
            calculated (not fully implemented). The default is False.

        Returns
        -------
        ref : pointCollection.data
            data corresponding to the reference track (1 column per cycle, one row for each crossing track)
        crossing: pointCollection.data
            data corresponding to the crossing tracks (one row for each crossing track.
        delta: pointColleciton.data, output only if calc_delta is True
            Differences between crossing tracks and reference track (same shape as crossing)
        '''


        try:
            rgt=self.attrs['ReferenceGroundTrack']
        except KeyError:
            rgt=self.track_num
        xo={'ref':{},'crossing':{},'both':{}}
        n_cycles=self.ROOT.h_corr.shape[1]
        zz=np.zeros(n_cycles)

        both_fields = ['delta_time','h_corr','h_corr_sigma','h_corr_sigma_systematic',
                      'ref_pt','rgt','latitude','longitude',
                      'cycle_number', 'dh_geoloc','tide_ocean','dac']
        ref_fields = ['x_atc','y_atc','fit_quality', 'atl06_quality_summary_zero_count',
                      'dem_h', 'geoid_h', 'fit_quality','n_slope','e_slope']
        xo_fields = ['beam_pair', 'atl06_quality_summary','spot','segment_id', 'along_track_rss']
        for field in both_fields+ref_fields:
            xo['ref'][field]=[]
        for field in both_fields + xo_fields:
            xo['crossing'][field]=[]

        for i1, ref_pt in enumerate(self.crossing_track_data.ref_pt):
            i0=np.flatnonzero(self.ROOT.ref_pt==ref_pt)[0]
            # fill vectors
            for field in ['latitude','longitude']:
                xo['ref'][field] += [getattr(self.ROOT, field)[i0] + zz]
            xo['ref']['ref_pt'] += [self.ROOT.ref_pt[i0]+zz]
            # set the crossing_track fields
            for field in both_fields + xo_fields:
                xo['crossing'][field] += [getattr(self.crossing_track_data, field)[i1]]

            # fill vectors for each cycle
            for field in ['delta_time', 'h_corr','h_corr_sigma','h_corr_sigma_systematic']:  # vars that are N_pts x N_cycles
                xo['ref'][field] += [getattr(self.ROOT, field)[i0,:]]
            xo['ref']['atl06_quality_summary_zero_count'] += [self.cycle_stats.atl06_summary_zero_count[i0,:]]
            xo['ref']['cycle_number'] += [getattr(self.ROOT,'cycle_number')]
            for field in ['dh_geoloc','tide_ocean','dac']:
                xo['ref'][field] += [getattr(self.cycle_stats, field)[i0,:]]
            # fields from ref_surf
            for field in ['x_atc','y_atc', 'fit_quality','n_slope','e_slope','dem_h','geoid_h']:
                xo['ref'][field] += [getattr(self.ref_surf, field)[i0] + zz]
        for field in xo['crossing']:
            xo['crossing'][field]=np.array(xo['crossing'][field])
        for field in xo['ref']:
            xo['ref'][field]=np.c_[xo['ref'][field]]

        ref=pc.data(columns=n_cycles).from_dict(xo['ref'])
        crossing=pc.data().from_dict(xo['crossing'])
        ref.assign(rgt = rgt + np.zeros(ref.shape, dtype=int))
        ref.assign(beam_pair = self.beam_pair + np.zeros(ref.shape, dtype=int))

        if calc_delta:
            delta={}
            delta['h_corr']=crossing.h_corr[:, None]-ref.h_corr
            delta['delta_time']=crossing.delta_time[:, None]-ref.delta_time
            delta['h_corr_sigma']=np.sqrt(crossing.h_corr_sigma[:, None]**2+ref.h_corr_sigma**2)
            delta['latitude']=ref.latitude.copy()
            delta['longitude']=ref.longitude.copy()
            delta=pc.data().from_dict(delta)
            return ref, crossing, delta
        else:
            return ref, crossing

    def plot(self):
        # method to plot the results.  At present, this plots corrected h AFN of x_atc
        n_cycles=self.ROOT.h_corr.shape[1]
        HR=np.nan+np.zeros((n_cycles, 2))
        h=list()
        #plt.figure(1);plt.clf()
        for cycle in range(self.cycles[1]+1- self.cycles[0], dtype=int):
            xx=self.ref_surf.x_atc
            zz=self.ROOT.h_corr[:,cycle]
            ss=self.ROOT.h_corr_sigma[:,cycle]
            good=np.abs(ss)<15
            ss[~good]=np.nan
            zz[~good]=np.nan
            if np.any(good):
                h0=plt.errorbar(xx[good],zz[good],ss[good], marker='o',picker=5)
                h.append(h0)
                HR[cycle,:]=np.array([zz[good].min(), zz[good].max()])
                #plt.plot(xx[good], zz[good], 'k',picker=None)
        temp=self.ROOT.h_corr.copy()
        temp[self.ROOT.h_corr_sigma>20]=np.nan
        temp=np.nanmean(temp, axis=1)
        plt.plot(xx, temp, 'k.', picker=5)
        plt.ylim((np.nanmin(HR[:,0]),  np.nanmax(HR[:,1])))
        plt.xlim((np.nanmin(xx),  np.nanmax(xx)))
        return h

    def from_ATL06(self, D6, tile_dirs=None, beam_pair=1, cycles=[1, 12],\
                   ref_pt_numbers=None, ref_pt_x=None, hemisphere=-1,\
                   mission_time_bds=None, max_xover_latitude=90, \
                   atc_shift_table=None,\
                   hold_list=None,\
                   verbose=False, DOPLOT=None,DEBUG=None, return_list=True):
        """
        Fit a collection of ATL06 files with ATL11 surface models

        Positional input:
            ATL06_files:  List of ATL06 files (from the same rgt)
            Required keyword inputs:
                beam_pair: beam pair for the current fit (default=1)
                cycles: cycles to be included in the current fit (default=2)
                tile_dirs: directories in which to find ATL06 data for crossovers
                hemisphere: +1 (north) or -1 (south), used to choose a projection
            Optional keyword arguments (not necessarily independent)
                max_xover_latitude: calculate crossovers for latitudes lower than this value
                mission_time_bds: starting and ending times for the mission
                verbose: write fitting info to stdout if true
                atc_shift_table: dict giving the along-track bias correction as a function of time
                hold_list: list of cycle, rgt, subproduct combinations to exclude
                DOPLOT: list of plots to make
                DEBUG: output debugging info
        """

        params_11=ATL11.defaults()
        if mission_time_bds is None:
            mission_time_bds=np.array((cycles[0]+2, cycles[1]+3))*91*24*3600

        # hard code the bin size until there's a good reason to change it
        index_bin_size=1.e4

        # setup the EPSG
        if hemisphere==1:
            params_11.EPSG=3413
        else:
            params_11.EPSG=3031

        # initialize the xover data cache
        D_xover_cache={}

        # loop over reference points
        P11_list=list()

        # New feature: use numpy's searchsorted to find the indexes for the current ref_pt
        # sort the input ATL06 by segment id, keep the list of seg_id values
        D6_sort_ind = np.argsort(np.nanmax(D6.segment_id, axis=1))
        D6=D6[D6_sort_ind]
        segid_sorted = np.nanmax(D6.segment_id, axis=1)
        start_segid = np.searchsorted(segid_sorted, ref_pt_numbers-params_11.N_search, side='left')
        end_segid = np.searchsorted(segid_sorted, ref_pt_numbers+params_11.N_search, side='right')

        D6_xyB = make_ATL06_xy_bins(D6, 100)

        xover_tile_schemas={}
        for ref_pt, x_atc_ctr, i0, i1 in zip(ref_pt_numbers, ref_pt_x, start_segid, end_segid):

            # section 5.1.1
            D6_sub=D6[i0:i1]

            keep = (D6_sub.h_li > params_11.min_valid_h) & (D6_sub.h_li < params_11.max_valid_h)
            if np.any(keep==0):
                D6_sub.h_li[keep==0]=np.nan
                D6_sub=D6_sub[np.any(np.isfinite(D6_sub.h_li), axis=1)]

            if D6_sub.h_li.shape[0]<=1:
                if verbose:
                    print("not enough data at ref pt=%d" % ref_pt)
                continue

            # fix for bad regressions in x_atc for latitude: reject if there are not two distinct x_atc values
            if np.nanmax(D6_sub.x_atc)-np.nanmin(D6_sub.x_atc) < 20:
                if verbose:
                    print("not enough distinct x_atc values at ref_pt=%d" % ref_pt)
                continue

            #2a. define representative x and y values for the pairs
            pair_data=ATL06_pair(D6_sub)
            if ~np.any(np.isfinite(pair_data.y)):
                continue
            P11=ATL11.point(N_pairs=len(pair_data.x), rgt=D6_sub.rgt[0, 0],\
                            ref_pt=ref_pt, beam_pair=D6_sub.BP[0, 0],  \
                            x_atc_ctr=x_atc_ctr, \
                            track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),\
                            cycles=cycles,\
                            mission_time_bds=mission_time_bds)

            P11.DOPLOT=DOPLOT
            # step 2: select pairs, based on reasonable slopes
            try:
                P11.select_ATL06_pairs(D6_sub, pair_data)
            except np.linalg.LinAlgError:
                if verbose:
                    print("LinAlg error in select_ATL06_pairs ref pt=%d" % ref_pt)
            #if P11.ref_surf.complex_surface_flag:
            #    P11.select_ATL06_pairs(D6_sub, pair_data, complex_surface_flag=True)

            if P11.ref_surf.fit_quality > 0:
                #P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.quality_summary, ref_pt))
                continue

            if np.sum(P11.valid_pairs.all) < 2:
                continue

            # select the y coordinate for the fit (in ATC coords)
            P11.select_y_center(D6_sub, pair_data)

            if np.sum(P11.valid_pairs.all) < 2:
                continue

            if P11.ref_surf.fit_quality > 0:
                #P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.quality_summary, ref_pt))
                continue

            # regress the geographic coordinates from the data to the fit center
            P11.ROOT.latitude, P11.ROOT.longitude = regress_to(D6_sub,\
                            ['latitude','longitude'], ['x_atc','y_atc'],
                            [x_atc_ctr, P11.y_atc_ctr])

            # find the reference surface
            P11.find_reference_surface(D6_sub, pair_data)

            if 'inversion failed' in P11.status:
                #P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.fit_quality, ref_pt))
                continue
            # get the slope and curvature parameters
            P11.characterize_ref_surf()

            # correct the heights from other cycles to the reference point using the reference surface
            P11.corr_heights_other_cycles(D6_sub)

            invalid_h = (P11.ROOT.h_corr < params_11.min_valid_h) | (P11.ROOT.h_corr > params_11.max_valid_h)
            if np.any(invalid_h):
                P11.ROOT.h_corr[invalid_h] = np.nan
                P11.ROOT.h_corr_sigma[invalid_h] = np.nan
                P11.ROOT.h_corr_sigma_systematic[invalid_h] = np.nan

            P11.ROOT.quality_summary = np.logical_not(
                    (P11.cycle_stats.min_signal_selection_source <=1) &\
                    (P11.cycle_stats.min_snr_significance < 0.02) &\
                    (P11.cycle_stats.atl06_summary_zero_count > 0) )

            # find the center of the bin in polar stereographic coordinates
            x0, y0=regress_to(D6_sub, ['x','y'], ['x_atc', 'y_atc'], [x_atc_ctr,P11.y_atc_ctr])

            # get the DEM elevation
            P11.ref_surf.dem_h=regress_to(D6_sub,
                                          ['dem_h'], ['x_atc', 'y_atc'],
                                          [x_atc_ctr,P11.y_atc_ctr])
            # Temporary rel007 bug fix
            if (P11.ref_surf.dem_h < params_11.min_valid_h) | (P11.ref_surf.dem_h > params_11.max_valid_h):
                print(f"\t ATL11/data.py: found ref_surf.dem_h of {P11.ref_surf.dem_h} at lat, lon = {np.nanmean(D6_sub.latitude)} {np.nanmean(D6_sub.longitude)}")
                print(f"\t\t setting dem_h to invalid for pair {P11.beam_pair} reference point {P11.ref_pt}")
                P11.ref_surf.dem_h = np.nan

            # get the geoid height
            P11.ref_surf.geoid_h=regress_to(D6_sub,
                                          ['geoid_h'], ['x_atc', 'y_atc'],
                                          [x_atc_ctr,P11.y_atc_ctr])
            # get the geoid conversion
            P11.ref_surf.geoid_free2mean=regress_to(D6_sub,
                                          ['geoid_free2mean'], ['x_atc', 'y_atc'],
                                          [x_atc_ctr,P11.y_atc_ctr])
            # get the data for the crossover point
            if tile_dirs is not None and np.abs(P11.ROOT.latitude) < max_xover_latitude:
                D_xover=ATL11.get_xover_data(x0, y0, P11.rgt,
                                             D_xover_cache, index_bin_size, params_11,
                                             xy_bin=D6_xyB,
                                             tile_dirs=tile_dirs,
                                             hold_list=hold_list,
                                             schemas=xover_tile_schemas,
                                             verbose=verbose)
                P11.corr_xover_heights(D_xover, atc_shift_table=atc_shift_table)
            # if we have read any data for the current bin, run the crossover calculation
            PLOTME=False
            if PLOTME:
                plt.figure()
                for key in D_xover_cache.keys():
                    plt.plot(D_xover_cache[key]['D'].x, D_xover_cache[key]['D'].y,'k.')

                plt.plot(D_xover.x, D_xover.y,'m.')
                plt.plot(x0, y0,'g*')

            if not np.isfinite(P11.ROOT.latitude):
                continue
            P11_list.append(P11)

        if return_list:
            return P11_list

        if len(P11_list) > 0:
            cycles=[np.nanmin([Pi.cycles for Pi in P11_list]), np.nanmax([Pi.cycles for Pi in P11_list])]
            N_coeffs=np.nanmax([Pi.N_coeffs  for Pi in P11_list])
            return ATL11.data(track_num=P11_list[0].rgt, beam_pair=beam_pair, cycles=cycles, N_coeffs=N_coeffs, N_pts=len(P11_list)).from_list(P11_list)
        else:
            return None


def make_ATL06_xy_bins(D6, bin_size):
    '''
    Make an array that covers all the rounded locations in D6, plus one bin on either side
    '''

    xy0 = pc.unique_by_rows(np.round(np.c_[D6.x.ravel(), D6.y.ravel()]/bin_size)*bin_size)
    dx, dy = np.meshgrid(np.array([-1, 0, 1])*bin_size, np.array([-1, 0, 1])*bin_size)
    xy0 = pc.unique_by_rows(np.c_[(xy0[:,0][:, None]+dx.ravel()[None,:]).ravel(), (xy0[:,1][:, None]+dy.ravel()[None,:]).ravel()])
    return xy0

def shift_lon(lon, lon0=0):
    """
    wrap longitudes to +-180
    """
    lon += lon0
    lon[lon>180] -= 360
    lon[lon<-180] += 360
    return lon

def regress_to(D, out_field_names, in_field_names, in_field_pt, DEBUG=None):

    """
    Regress a series of data values to a reference point

    inputs:
        D: data structure
        out_field_names: list of field names for which we are trying to recover values
        in_field_names: list of field names as independent variables in the regression
        in_field_pt: location of the regression center (in variables in_field_names)
    """

    D_in = np.array(tuple([getattr(D, name).ravel() for name in in_field_names])).T
    D_out = np.array(tuple([getattr(D, name).ravel() for name in out_field_names])).T
    if 'longitude' in out_field_names:
        # if longitude is in the regression parameters, need to unwrqp it
        lon_col=out_field_names.index('longitude')
        lon0=np.nanmedian(D_out[:,lon_col])
        D_out[:, lon_col]=shift_lon(D_out[:, lon_col], lon0=-lon0)
    good_rows=np.all(~np.isnan( np.concatenate( (D_in,D_out), axis=1)), axis=1)

    if np.sum(good_rows) < 2:
        return np.zeros(len(out_field_names))+np.nan

    # build the regression matrix
    G=np.ones((np.sum(good_rows),len(in_field_names)+1) )
    for k in range(len(in_field_names)):
        G[:,k+1] = D_in[good_rows,k] - in_field_pt[k]

    # calculate the regression coefficients (the intercepts will be the first row)
    out_pt = np.linalg.lstsq(G,D_out[good_rows,:], rcond=None)[0][0,:]
    if 'longitude' in out_field_names:
        out_pt[lon_col] = shift_lon([out_pt[lon_col]], lon0=lon0)[0]
    if DEBUG is not None:
        print('lat_ctr,lon_ctr',out_pt)

    return out_pt
