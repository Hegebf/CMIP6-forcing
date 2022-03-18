import numpy as np
import xarray as xr
import pandas as pd
import intake
import json

def area_weights(lat_bnds, lon_bnds): 
    # computes exact area weigths assuming earth is a perfect sphere
    lowerlats = np.radians(lat_bnds[:,0]); upperlats = np.radians(lat_bnds[:,1])
    difflon = np.radians(np.diff(lon_bnds[0,:])) 
    # if the differences in longitudes are all the same
    areaweights = difflon*(np.sin(upperlats) - np.sin(lowerlats));
    areaweights /= areaweights.mean()
    return areaweights # list of weights, of same dimension as latitude

def areacella_weights(model):
    # computes normalized area-weights from piControl areacella 
    # (which I think should be the same as for other experiements too from the same model):
    col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(col_url)
    if model == 'IPSL-CM5A2-INCA':
        # no areacella in cloud for piControl for this model
        area_cat = col.search(experiment_id = 'abrupt-4xCO2', source_id = model, variable_id=['areacella'])
    else:
        area_cat = col.search(experiment_id = 'piControl', source_id = model, variable_id=['areacella'])
    area_dset_dict = area_cat.to_dataset_dict(zarr_kwargs={'consolidated': True}, cdf_kwargs={'chunks': {}})
    
    for key in area_dset_dict.keys():
        area_ds = area_dset_dict[key]
    areas = area_ds['areacella'].values[0,:,0]
    norm_areas = areas/areas.mean()
    return norm_areas

def global_averaging(ds, varlist = ['tas', 'rlut', 'rsut', 'rsdt'], calendar = None):
    # function for computing global average
    # takes in xarray dataset for a single member with the four variabels [tas, rlut, rsut, rsdt]
    # returns global annual averages in pandas dataframe
    model = ds.source_id
    exp = ds.experiment_id
    if calendar == None:
        ds_calendar = ds.time.encoding['calendar'] 
        # this does not work if loading files with xr.mfopen_dataset and concatenating along time dimension.
        # https://github.com/pydata/xarray/issues/2436
    else:
        # So in that case the calendar must be supplied as input to this function instead
        ds_calendar = calendar
    
    firstmonth = ds.time.to_index().month[0] # not jan for IPSL-CM6A-LR members 2-12
    # annual average from feb-jan, march-feb, etc. when firstmonth is not jan. 
    print('first month of dataset is:', firstmonth)
    
    # make sure time dimension is a multiple of 12, and that values are removed in the beginning/end for some experiments
    if model == 'MRI-ESM2-0' and exp == 'abrupt-4xCO2':
        # note: piControl values in these time series before the branch time
        member = ds.variant_label
        rstring = member[1:3].replace('i','')
        firstmonth = int(rstring)
        if firstmonth>1: # delete values before 'firstmonth', and some values in the end:
            ds = ds.isel(time=np.arange(firstmonth-1,len(ds.time)-(12-firstmonth+1)))
            # check result
            print('firstmonth after removing piControl values is:', ds.time.to_index().month[0])
    elif model == 'CNRM-CM6-1' and exp == 'abrupt-4xCO2':
        # delete values in the end to get a multiple of 12 length for the annual mean computations
        ds = ds.isel(time=np.arange(len(ds.time)-(12-firstmonth+1)))
    elif model == 'CESM2' and exp == 'hist-GHG':
        member = ds.variant_label
        if member in ['r1i1p1f1']:
            ds = ds.isel(time=np.arange(0,len(ds.time)-6)) # make it stop in dec 2014
    elif model == 'EC-Earth3' and exp == 'historical':
        member = ds.variant_label
        if member in ['r11i1p1f1', 'r13i1p1f1', 'r15i1p1f1']:
            # remove the first december value
            ds = ds.isel(time=np.arange(1,len(ds.time)))
      
    
    
    # find normalized area weights to be used in the global averaging
    if model in ['CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']: 
        # French models missing lat_bnds and lon_bnds coordinates
        # compute areaweights using areacella
        area_w = areacella_weights(model)
    elif model in ['ICON-ESM-LR']:
        # this model has triangular grid cells with similar areas
        # Model description: https://doi.org/10.1029/2017MS001242
        # Older paper on this model: https://doi.org/10.5194/gmd-6-735-2013, saying that for the resolution used here (R2B4),
        # the ratio between the maximum and minimum cell areas is 1.38.
        # I don't think there is a systematic difference between high and low latitudes
        # grid weights constructed using cdo gridweights on the file tas_Amon_ICON-ESM-LR_piControl_r1i1p1f1_gn_400001-400912.nc
        print('area weigths constricted from cdo gridweights')
        gridweights = xr.open_dataset('../Processed_data/gridweights_ICON-ESM-LR.nc')
        areaweights = gridweights['cell_weights']
        areaweights /= areaweights.mean()
    else:
        # sometimes we need to get rid of a time dimension of the lat and lon bnds:
        # I think this is because the bnds are stored/interpreted as variables rather than coordinates
        
        lon_bnds = ds.lon_bnds.values
        lat_bnds = ds.lat_bnds.values
        # for some datasets we need to change these values:
        
        if model == 'NorCPM1' and exp == 'historical':
            member = ds.variant_label
            if member == 'r10i1p1f1':
                lat_bnds = ds.lat_bnds.values[0,:,:]
        elif exp == 'ssp585' and model == 'CESM2-WACCM':
            member = ds.variant_label
            if member == 'r1i1p1f1':
                lat_bnds = ds.lat_bnds.values[0,:,:]
        elif exp == 'ssp126' and model == 'CESM2-WACCM':
            member = ds.variant_label
            if member == 'r1i1p1f1':
                lat_bnds = ds.lat_bnds.values[:,:,0]
        elif model in ['MPI-ESM-1-2-HAM', 'EC-Earth3', 'EC-Earth3-AerChem'] and exp == 'piClim-control':
            lat_bnds = ds.lat_bnds.values[0,:,:]
            lon_bnds = ds.lon_bnds.values[0,:,:]
        elif model == 'EC-Earth3' and exp == 'historical':
            member = ds.variant_label
            if member == 'r3i1p1f1':
                lat_bnds = ds.lat_bnds.values[0,:,:]
                print(np.shape(lat_bnds), np.shape(lon_bnds))
        elif model == 'EC-Earth3' and exp == 'ssp245':
            member = ds.variant_label
            if member == 'r6i1p1f2':
                lat_bnds = ds.lat_bnds.values[0,:,:]
        #print(np.shape(lat_bnds), np.shape(lon_bnds))
        area_w = area_weights(lat_bnds, lon_bnds)
        #print(np.shape(area_w))
            
    day_weights = compute_day_weights(ds, calendar = ds_calendar, first_month = firstmonth)    
    for variable in varlist:
        print(variable)
        data = ds[variable]
        #print(data)
        #print(data.dims)

        # global average
        if model in ['ICON-ESM-LR']: # triangular grid cells (unstructured grid on a different format)
            area_avg = (data * areaweights).mean(dim = ['i']) # average over grid cell index (i)
        else:
            #print(data)
            area_avg = (data.transpose('time', 'lon', 'lat') * area_w).mean(dim=['lon', 'lat'])
        
        # annual average
        #print(area_w[:12])
        #print(area_avg.values[:12])
        day_weighted_avg = area_avg*day_weights
        
        if firstmonth == 1 or firstmonth == 13:
            annualmean = day_weighted_avg.groupby('time.year').mean('time')
            #print(annualmean)
            #print(annualmean.values)
            #print('annualmean.data_vars:', annualmean.data_vars)
            annualmean_pd = annualmean.to_pandas()
            df_col = pd.DataFrame(annualmean_pd, columns = [variable])
            
            #index = annualmean.year.values
        else: # this code is slow, but rarely needed
            annualmean_array = np.array([day_weighted_avg[i*12:(i+1)*12].mean() for i in range(int(len(day_weighted_avg)/12))])
            annualmean = xr.DataArray(annualmean_array)
            startyears = area_avg.time.to_index().year[0::12].values
            startmonths = area_avg.time.to_index().month[0::12].values
            endyears = area_avg.time.to_index().year[12-1::12].values
            endmonths = area_avg.time.to_index().month[12-1::12].values
            index = [str(startyears[t]) + str("%02d" % startmonths[t]) + '-' + str(endyears[t]) + str("%02d" % endmonths[t]) for t in range(len(startyears))]
            df_col = pd.DataFrame(annualmean.values, columns = [variable], index = index)
            
        if variable == varlist[0]:
            # create dataframe for storing all results
            df = df_col
        else:
            df = pd.merge(df, df_col, left_index=True, right_index=True, how='outer')
    return df

# function copied from: http://xarray.pydata.org/en/stable/examples/monthly-means.html
def leap_year(year, calendar='standard'):
    """Determine if year is a leap year"""
    leap = False
    if ((calendar in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year < 1583)):
            leap = False
    return leap

# function copied from: http://xarray.pydata.org/en/stable/examples/monthly-means.html
def get_dpm(time, calendar='standard'):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar) and month == 2:
            month_length[i] += 1
    return month_length

# days per month:
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # I assume this is the same as noleap
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'julian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], ##### I think this should be correct
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
      }



def compute_day_weights(ds, calendar = 'noleap', first_month = 1): # new function
    month_length = xr.DataArray(get_dpm((ds.time.to_index()), calendar=calendar), coords=[ds.time], name='month_length')
    if first_month == 1:
        norm_by_annual = month_length.groupby('time.year').mean('time') # make annual mean
        norm_by_monthly = np.concatenate([np.tile(norm_by_annual.values[i], 12) for i in range(len(norm_by_annual.values))])
        #print(len(month_length), len(norm_by_annual), len(norm_by_monthly)) ### useful for debugging
    else: 
        norm_by_annual = np.array([month_length[i*12:(i+1)*12].mean() for i in range(int(len(month_length)/12))])
        norm_by_monthly = np.concatenate([np.tile(norm_by_annual[i], 12) for i in range(len(norm_by_annual))]) 
    weights = month_length/norm_by_monthly
    # normalized to have mean 1
    return weights

sepvar_dict = {'piControl': 
               {'EC-Earth3': 
                ['r2i1p1f1']},
               'historical':
               {'NorCPM1':
               ['r11i1p1f1','r12i1p1f1','r13i1p1f1','r14i1p1f1','r15i1p1f1',
                'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r1i1p1f1', 
                'r20i1p1f1', 'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1',
                'r25i1p1f1', 'r27i1p1f1', 'r28i1p1f1', 'r29i1p1f1', 'r2i1p1f1',
                'r30i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1', 'r6i1p1f1',
                'r7i1p1f1', 'r8i1p1f1', 'r9i1p1f1']},
                'ssp585': 
                {'ACCESS-CM2':
                [' r1i1p1f1']}}

def sepvar_test(exp, model, member):
    if exp in sepvar_dict:
        if model in sepvar_dict[exp]:
            if member in sepvar_dict[exp][model]:
                return True
            
def ds_cloud(exp, model, member, var_list = ['tas', 'rlut', 'rsut', 'rsdt']):
    col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(col_url)
    col_member = col.search(experiment_id = exp, source_id = model, member_id = member, variable_id=var_list, table_id='Amon') 
    #if model == 'EC-Earth3' and exp == 'piControl' and member == 'r2i1p1f1':
    #    # the time dimension of the different variables seems to have slightly different values, e.g jan-15 vs jan-16. This dimension is therefore too long
    #    col_member.aggregation_info['aggregations'][0].update({'options': {'compat': 'override'}})
    #    dset_dict = col_member.to_dataset_dict(zarr_kwargs={'consolidated': True}, cdf_kwargs={'chunks': {}})
    #else:
    dset_dict = col_member.to_dataset_dict(zarr_kwargs={'consolidated': True}, cdf_kwargs={'chunks': {}})
    if len(dset_dict.keys()) > 1:
        print(dset_dict.keys())
        print('Too many keys. This should normally not happen')
        print('But it happens for some data that are stored in both CMIP and DAMIP, when they should be only in DAMIP')
        for key in dset_dict.keys():
            if 'DAMIP' in key:
                ds = dset_dict[key]
    else:
        for key in dset_dict.keys():
            ds = dset_dict[key]
    ds = ds.sel(member_id = member)
    return ds

def global_average_sepvar(exp, model, member, var_list = ['tas', 'rlut', 'rsut', 'rsdt']):
    for var in var_list:
        ds_var = ds_cloud(exp, model, member, var_list = [var])
        avg_df_var = global_averaging(ds_var, varlist = [var])
        if var == var_list[0]:
            # create dataframe for storing all results
            avg_df = avg_df_var
        else:
            avg_df = pd.merge(avg_df, avg_df_var, left_index=True, right_index=True, how='outer')
    return avg_df

sepvar_dict_downloadeddata = {'piControl': 
                               {'EC-Earth3-CC': ['r1i1p1f1']},
                              'ssp126':
                               {'KIOST-ESM': ['r1i1p1f1']},
                              'ssp245':
                               {'EC-Earth3': ['r20i1p1f1', 'r6i1p1f2']}
                              }

def global_average_sepvar_downloadeddata(exp, model, member, paths, calendar, var_list = ['tas', 'rlut', 'rsut', 'rsdt']):
    for var in var_list:
        print(var)
        model_exp_member_var_paths = []
        for path in paths:
            filename = path.rsplit("/")[-1]
            if var in str(filename):
                 model_exp_member_var_paths.append(path)
        #ds_var = xr.open_mfdataset(model_exp_member_var_paths, combine='by_coords', join = 'override', concat_dim='time', parallel = True, preprocess = preprocess, use_cftime = True)
        ds_var = xr.open_mfdataset(model_exp_member_var_paths, combine='by_coords', join = 'exact', concat_dim='time', parallel = True, preprocess = preprocess, use_cftime = True)
        #ds_var = ds_var.drop_vars('time_bnds')
        #print(ds_var)
        avg_df_var = global_averaging(ds_var, varlist = [var], calendar = calendar)
        
        if var == var_list[0]:
            # create dataframe for storing all results
            avg_df = avg_df_var
        else:
            avg_df = pd.merge(avg_df, avg_df_var, left_index=True, right_index=True, how='outer')
    return avg_df

def preprocess(ds):
    # convert lat_bnds and lon_bnds to coordinates if they are variables
    data_vars = list(ds.keys())
    if 'lat_bnds' in data_vars:
        ds = ds.set_coords('lat_bnds')
    if 'lon_bnds' in data_vars:
        ds = ds.set_coords('lon_bnds')
    return ds

def filecheck(filestr, all_files):
    for filename in all_files:
        if filestr in str(filename):
            print('At least one file exists already, check manually if we have downloaded all')
            return True
        
def missing_esgf_test(exp, model, member):
    with open('../Data_availability/ESGF/esgf-missing_march3rd.json', 'r') as f:
        missing_dict = json.load(f)
    if exp in missing_dict:
        if model in missing_dict[exp]:
            if member in missing_dict[exp][model]:
                return True
            
