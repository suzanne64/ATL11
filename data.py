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
from PointDatabase.ATL06_data import ATL06_data



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


    def get_xovers(self, rgt, pair):
        xo={'ref':{},'crossing':{},'both':{}}
        for field in ['time','h','h_sigma','ref_pt_number','rgt','PT','atl06_quality_summary','latitude','longitude']:
            xo['ref'][field]=[]
            xo['crossing'][field]=[]
        xo['crossing']['RSSz']=[]

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
                xo['ref']['atl06_quality_summary'] += [self.cycle_stats.ATL06_summary_zero_count[i0, ic] == 0]

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
        for field in xo['crossing']:
            xo['crossing'][field]=np.array(xo['crossing'][field])
        for field in xo['ref']:
            xo['ref'][field]=np.array(xo['ref'][field])
        ref=point_data().from_dict(xo['ref'])
        crossing=point_data().from_dict(xo['crossing'])
        delta={}
        delta['h']=crossing.h-ref.h
        delta['time']=crossing.time-ref.time
        delta['sigma_h']=np.sqrt(crossing.h_sigma**2+ref.h_sigma**2)
        delta['latitude']=ref.latitude.copy()
        delta['longitude']=ref.longitude.copy()
        delta=point_data().from_dict(delta)
        return ref, crossing, delta



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

    def from_ATL06(self, ATL06_files, beam_pair=1, N_cycles=2,  output_file=None, GI_file=None, ref_pt_numbers=None, num_ref_pts=None, first_ref_pt=None, last_ref_pt=None,  mission_time_bds=None, lonlat_bounds=None, verbose=False, DOPLOT=None, DEBUG=None):
        """
        Fit a collection of ATL06 files with ATL11 surface models

        Positional input:
            ATL06_files:  List of ATL06 files (from the same rgt)
            Required keyword inputs:
                beam_pair: beam pair for the current fit (default=1)
                N_cycles: Number of cycles in the current fit (default=2)
                output_file: File into which to save the ouput
                GI_file: geo_index file from which to read ATL06 data
            Optional keyword arguments (not necessarily independent)
                ref_pt_numbers: Specific reference point numbers to fit
                first_ref_pt: first reference point to attempt to fit
                num_ref_pts: number of reference points to include in the fit
                last_ref_pt: last reference point to include in the fit
                mission_time_bds: starting and ending times for the mission
                lonlat_bds: region within which to fit points
                verbose: write fitting info to stdout if true
                DOPLOT: list of plots to make
                DEBUG: output debugging info
        """

        params_11=ATL11.defaults()
        seg_number_skip=int(params_11.seg_atc_spacing/20);
        if mission_time_bds is None:
            mission_time_bds=np.array([286.*24*3600, 398.*24*3600])

        # read in the ATL06 data from all the repeats
        D6_list=[]
        for filename in ATL06_files:
            try:
                D6_list.append(ATL06_data(field_dict=params_11.ATL06_field_dict, beam_pair=beam_pair).from_file(filename))
            except KeyError:
                pass
        if len(D6_list)==0:
            return None
        D6=ATL06_data(beam_pair=beam_pair).from_list(D6_list)

        if lonlat_bounds is not None:
            keep = (D6.longitude >= lonlat_bounds[0])
            keep &= (D6.latitude >= lonlat_bounds[1])
            keep &= (D6.longitude <= lonlat_bounds[2])
            keep &= (D6.latitude <= lonlat_bounds[3])
            keep = np.any(keep, axis=1)
            if not np.any(keep):
                return None
        D6.index(keep)

        # reorder data rows from D6 by cycle
        D6.index(np.argsort(D6.cycle_number[:,0],axis=0))

        # choose the hemisphere and project the data to polar stereographic
        if np.max(D6.latitude) < 0:
            D6.get_xy(None, EPSG=3031)
            params_11.EPSG=3031
            index_bin_size=1.e4
        else:
            D6.get_xy(None, EPSG=3413)
            params_11.EPSG=3413
            index_bin_size=1e4

        # get list of reference points
        if ref_pt_numbers is None:
            uId, iId=np.unique(D6.segment_id.ravel(), return_index=True)
            ctrSegs=np.mod(uId, seg_number_skip)==0
            ref_pt_numbers=uId[ctrSegs]
            ref_pt_x=D6.x_atc.ravel()[iId[ctrSegs]]
        else:
            ref_pt_x=ref_pt_numbers*20

      # apply input arguments to the input reference points
        if first_ref_pt is not None:
            these=ref_pt_numbers>=first_ref_pt
            ref_pt_numbers=ref_pt_numbers[these]
            ref_pt_x=ref_pt_x[these]

        if last_ref_pt is not None:
            these=ref_pt_numbers<=last_ref_pt
            ref_pt_numbers=ref_pt_numbers[these]
            ref_pt_x=ref_pt_x[these]

        if num_ref_pts is not None:
            ref_pt_numbers=ref_pt_numbers[0:int(num_ref_pts)]
            ref_pt_x=ref_pt_x[0:int(num_ref_pts)]

        # initialize the xover data cache
        D_xover_cache={}

        last_count=0
        # loop over reference points
        P11_list=list()
        for count, ref_pt_number in enumerate(ref_pt_numbers):

            x_atc_ctr=ref_pt_x[count]
            # section 5.1.1
            D6_sub=D6.subset(np.any(np.abs(D6.segment_id-ref_pt_number) <= params_11.N_search, axis=1), by_row=True)
            if D6_sub.h_li.shape[0]<=1:
                if verbose:
                    print("not enough data at ref pt=%d" % ref_pt_number)
                continue

            #2a. define representative x and y values for the pairs
            pair_data=D6_sub.get_pairs(datasets=['x_atc','y_atc','delta_time','dh_fit_dx','dh_fit_dy','segment_id','cycle_number','h_li'])   # this might go, similar to D6_sub
            if ~np.any(np.isfinite(pair_data.y)):
                continue
            P11=ATL11.point(N_pairs=len(pair_data.x), rgt=D6_sub.rgt[0, 0], ref_pt_number=ref_pt_number, pair_num=D6_sub.BP[0, 0],  x_atc_ctr=x_atc_ctr, track_azimuth=np.nanmedian(D6_sub.seg_azimuth.ravel()),N_cycles=N_cycles,  mission_time_bds=mission_time_bds )

            P11.DOPLOT=DOPLOT
            # step 2: select pairs, based on reasonable slopes
            P11.select_ATL06_pairs(D6_sub, pair_data)
            if P11.ref_surf.surf_fit_quality_summary > 0:
                P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
                continue

            # select the y coordinate for the fit (in ATC coords)
            P11.select_y_center(D6_sub, pair_data)
            if P11.ref_surf.surf_fit_quality_summary > 0:
                P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
                continue

            # regress the geographic coordinates from the data to the fit center
            P11.corrected_h.ref_pt_lat, P11.corrected_h.ref_pt_lon = ATL11.regress_to(D6_sub,['latitude','longitude'], ['x_atc','y_atc'], [x_atc_ctr, P11.y_atc_ctr])

            # find the reference surface
            P11.find_reference_surface(D6_sub)
            if 'inversion failed' in P11.status:
                P11_list.append(P11)
                if verbose:
                    print("surf_fit_quality=%d at ref pt=%d" % (P11.ref_surf.surf_fit_quality_summary, ref_pt_number))
                continue

            # correct the heights from other cycles to the reference point using the reference surface
            P11.corr_heights_other_cycles(D6_sub)

            # find the center of the bin in polar stereographic coordinates
            x0, y0=ATL11.regress_to(D6_sub, ['x','y'], ['x_atc', 'y_atc'], [x_atc_ctr,P11.y_atc_ctr])

            # get the data for the crossover point
            D_xover=ATL11.get_xover_data(x0, y0, P11.rgt, GI_file, D_xover_cache, index_bin_size, params_11)
            # if we have read any data for the current bin, run the crossover calculation
            PLOTME=False#isinstance(D_xover, point_data);
            if PLOTME:
                plt.figure()
                for key in D_xover_cache.keys():
                    plt.plot(D_xover_cache[key]['D'].x, D_xover_cache[key]['D'].y,'k.')

                plt.plot(D_xover.x, D_xover.y,'m.')
                plt.plot(x0, y0,'g*')

            P11.corr_xover_heights(D_xover)
            P11_list.append(P11)
            if count-last_count>500:
                print("completed %d segments, ref_pt_number= %d" %(count, ref_pt_number))
                last_count=count

        if len(P11_list) > 0:
            N_cycles=np.nanmax([Pi.N_cycles for Pi in P11_list])
            N_coeffs=np.nanmax([Pi.N_coeffs  for Pi in P11_list])
            return ATL11.data(track_num=P11_list[0].rgt, pair_num=pair, N_cycles=N_cycles, N_coeffs=N_coeffs, N_pts=len(P11_list)).from_list(P11_list)
        else:
            return ATL11.data(track_num=P11_list[0].rgt, pair_num=pair, N_cycles=N_cycles, N_coeffs=N_coeffs, N_pts=len(P11_list))
