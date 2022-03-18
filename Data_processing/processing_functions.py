# data processing functions

import pandas as pd
import os
import json
import xarray as xr
import intake
from global_annual_means import *
from matplotlib import pyplot as plt

col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(col_url)

idealised_exp = ['abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2']
hist_exp = ['historical', 'hist-GHG', 'hist-aer', 'hist-nat']
ssp_exp = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']


def find_members(model, exp, datatype = 'anomalies'):
    if datatype == 'anomalies':
        directory = '../Processed_data/Global_annual_anomalies/'
    elif datatype == 'means':
        directory = '../Processed_data/Global_annual_means_csv/'
    elif datatype == 'forcing':
        directory = '../Estimates/Transient_forcing_estimates/'
    modelexpdirectory = os.path.join(directory, model, exp)
    filenames = [f.name for f in os.scandir(modelexpdirectory) if f.name !='.ipynb_checkpoints']
    #if model == 'EC-Earth3' and exp == 'ssp245':
    #    print([print(file) for file in filenames])
        
    members = [file.rsplit('_')[2] for file in filenames if file != '.DS_Store']
    members.sort()
    return members

def make_exp_dict(directory = '../Processed_data/Global_annual_means_csv/'):
    experiments = {}
    model_names = [ f.name for f in os.scandir(directory) if f.is_dir() and f.name !='.ipynb_checkpoints']
    model_names.sort()

    for model in model_names:
        experiments[model] = {}
        modeldirectory = os.path.join(directory, model)
        modelexp_names = [ f.name for f in os.scandir(modeldirectory) if f.is_dir() and f.name !='.ipynb_checkpoints']
        for exp in modelexp_names:
            #print(exp)
            experiments[model][exp] = find_members(model, exp, datatype = 'means')
    return experiments

def can_load_from_cloud(exp, model, member):
    # Returns True if dataset can be loaded from cloud without errors:
    with open('../Data_availability/GoogleCloud/cloud-'+exp+'.json', 'r') as f:
        cloud_dict = json.load(f)
    with open('../Data_availability/GoogleCloud/cloud-buterrors.json', 'r') as f:
        error_dict = json.load(f)
    with open('../Data_availability/GoogleCloud/cloud-removed_march17.json', 'r') as f:
        removed_dict = json.load(f)
    if exp in error_dict:
        if model in error_dict[exp]:
            if member in error_dict[exp][model]:
                return False
    if exp in removed_dict:
        if model in removed_dict[exp]:
            if member in removed_dict[exp][model]:
                return False
    if model in cloud_dict:
        if member in cloud_dict[model]:
            return True
        
def load_ds(exp, model, member):
    # will be used just for finding the metadata, so we need only a single variable or file
    if can_load_from_cloud(exp, model, member) == True:
        cat = col.search(experiment_id = exp, variable_id='rlut', table_id='Amon',\
                         source_id = model, member_id = member)
        if len(cat) == 0:
            # then try another variable
            var = 'tas'
            print('The variable', var, 'has been loaded instead of rlut')
            cat = col.search(experiment_id = exp, variable_id=var, table_id='Amon',\
                         source_id = model, member_id = member)
        dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True},\
                                        cdf_kwargs={'chunks': {}})
        for key in dset_dict.keys():
            ds = dset_dict[key]
    else: # use downloaded files:
        exp_model_dir = '/Users/hege-beatefredriksen/Desktop/CMIP6_downloads/'\
        + exp + '_data/' + model + '/'
        model_exp_files = [f.name for f in os.scandir(exp_model_dir) if f.name[-3:] =='.nc']
        member_files = [file for file in model_exp_files if member in file]
        memberfile0_path = exp_model_dir + '/' + member_files[0]
        # try to open only first file, since it should contain what we need:
        ds = xr.open_dataset(memberfile0_path)
    return ds

branch_column_names = ['model', 'exp', 'member', 'parent_experiment_id', 'parent_variant_id', 
                    'branch_time_in_child', 'branch_time_in_parent', 'parent_time_units', 'branch_method']

calendar_column_names = ['model', 'exp', 'member', 'calendar']

def calendarinfo_df(model, modelexp_toload, load_also_downloaded_files = True):  
    df = pd.DataFrame(columns = calendar_column_names)

    for exp in modelexp_toload:
        for member in modelexp_toload[exp]:
            print(model, exp, member, 'will be loaded')
            if load_also_downloaded_files == False:
                if can_load_from_cloud(exp, model, member) != True:
                    print(model, exp, member, 'will not been loaded, because it is not in cloud')
                    continue
            ds = load_ds(exp, model, member)
            calendar = ds.time.encoding['calendar']
            
            df_row = pd.DataFrame([[model, exp, member, calendar]], columns = calendar_column_names)
            df = pd.concat([df, df_row], ignore_index = True)
    return df

def branchinfo_df(model, modelexp_toload, load_also_downloaded_files = True):
    df = pd.DataFrame(columns = branch_column_names)
    for exp in modelexp_toload:
        for member in modelexp_toload[exp]:
            print(model, exp, member, 'will be loaded')
            if load_also_downloaded_files == False:
                if can_load_from_cloud(exp, model, member) != True:
                    print(model, exp, member, 'will not been loaded, because it is not in cloud')
                    continue
            ds = load_ds(exp, model, member)
            #if hasattr(ds, 'parent_experiment_id'):
            #    parent_experiment_id = ds.parent_experiment_id
            #else: 
            #    parent_experiment_id = 'not an attribute'
            try:
                parent_experiment_id = ds.parent_experiment_id
                parent_variant_label = ds.parent_variant_label
                branch_time_in_child = ds.branch_time_in_child
                branch_time_in_parent = ds.branch_time_in_parent
                parent_time_units = ds.parent_time_units
                branch_method = ds.branch_method
                if model == 'EC-Earth3':
                    branch_time_in_child = str(branch_time_in_child).replace('D', '')
                    branch_time_in_parent = str(branch_time_in_parent).replace('D', '')
                print(model, exp, member, parent_experiment_id, parent_variant_label,\
                      branch_time_in_child, branch_time_in_parent, parent_time_units, branch_method)
                df_row = pd.DataFrame([[model, exp, member, parent_experiment_id, \
                                        parent_variant_label, branch_time_in_child,\
                                        branch_time_in_parent, parent_time_units, \
                                        branch_method]] ,columns = branch_column_names)
            except AttributeError: 
                # missing branch info for the historical EC-Earth3 experiments starting in 1970
                # From their paper https://doi.org/10.5194/gmd-14-4781-2021:
                # (Check also their table 1)
                # To create the set of initial conditions for SMHI-LENS we
                # start from six members (r1-3, r7-9) of the historical experiment for CMIP6 that was done with EC-Earth3-Veg. From
                # each of the six members we branch off breeding simulations
                # on 1 January 1970. These six breeding simulations are each
                # run for 20 years with constant forcing. From these six breeding runs we then select 50 initial states for the atmosphere
                # and the ocean as initial conditions for the large ensemble (Table 1). The initial date for each member of the large ensemble
                # is set to 1 January 1970.
                print(model, exp, member, 'is missing one or more attributes')
                df_row = pd.DataFrame([[model, exp, member, '-', '-', '-', '-', '-', '-']], columns = branch_column_names)
            df = pd.concat([df, df_row], ignore_index = True)
    return df
    
def data_to_load(model, experiments, previously_read):
    # used when some data exist already
    modelexp_toload = {}    

    # does previously_read cover all available experiments and members?
    for exp in experiments[model]:
        if exp in previously_read['exp'].values:  
            # then at least one member has been loaded before
            previously_read_exp = previously_read[exp == previously_read['exp']] 
            # dataframe reduced to only this experiment
            for member in experiments[model][exp]:
                if member in previously_read_exp['member'].values:
                    print(model, exp, member, ' exists')
                else:
                    if exp not in modelexp_toload:
                        modelexp_toload[exp] = [member] 
                        # needs to be tested for a model with new members
                    else:
                        modelexp_toload[exp].append(member)
                    #print('other members of', exp, 'have been looked at before, but member ', member, 'needs to be loaded')
        else:
            #print(exp,  ' is not available in file and needs to be loaded for all members', experiments[model][exp])
            modelexp_toload[exp] = experiments[model][exp]
            #newfile = True

    return modelexp_toload 


def load_anom(model, exp, member, length_restriction = None):
    #filename = model + '_' + exp + '_' + member + '_anomalies.txt'
    #file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
    filename = model + '_' + exp + '_' + member + '_anomalies.csv'
    file = os.path.join('../Processed_data/Global_annual_anomalies_csv/', model, exp, filename)
    
    data = pd.read_table(file, index_col=0, sep = ',')
    if model != 'AWI-CM-1-1-MR':
        data = data.dropna().reset_index()
    if length_restriction != None:
        data = data[:length_restriction]
    return data
    
def branch_info_corrections(branch_info_table):
    model = branch_info_table['model'][0] # all elements are the same, so just take the first
    experiments = list(branch_info_table['exp'].unique())
    for exp in experiments:
        exp_df = branch_info_table.loc[branch_info_table['exp'] == exp]
        members = list(exp_df['member'].unique())
        for member in members:
            member_df = exp_df.loc[exp_df['member'] == member]
            ind = member_df.index.values[0] # index of row
            
            # check that parent member exists:
            # if not, correct parent info if possible
            parent_exp = member_df['parent_experiment_id'].values[0]
            parent_member = member_df['parent_variant_id'].values[0]
            parent_exp_df = branch_info_table.loc[branch_info_table['exp'] == parent_exp]
            parent_member_df = parent_exp_df.loc[parent_exp_df['member'] == parent_member]
            if parent_member_df.empty and exp != 'piControl':
                print('Parent', parent_exp, parent_member, 'does not exist for:', exp, member)
                if len(parent_exp_df) == 1:
                    alt_parent_member = parent_exp_df['member'].values[0]
                    branch_info_table.at[ind, 'parent_variant_id'] = alt_parent_member
                    print(' - parent member info changed to the only other option available:', alt_parent_member)
                else:
                    print('Please tell this function how to select a parent member from', '\n', \
                    'the other alternative parent members available:', parent_exp, parent_exp_df['member'].values)
                    if model == 'IPSL-CM6A-LR' and parent_exp == 'piControl':
                        alt_parent_member = 'r1i1p1f1'
                    
                    print('Member', alt_parent_member, 'has been manually selected')
                    branch_info_table.at[ind, 'parent_variant_id'] = alt_parent_member
            if exp in ssp_exp:
                # ssp member should likely be the same as historical member
                if member != parent_member:
                    if member in parent_exp_df['member'].values:
                        branch_info_table.at[ind, 'parent_variant_id'] = member
                        print('Metadata said parent of', exp, member, 'should be', parent_exp, parent_member)
                        print('This parent member is changed for to match the child member')
                        
    return branch_info_table # with corrections  

def dpy(start_year, end_year, ds_calendar): # days per year
    leap_boolean = [leap_year(year, calendar = ds_calendar)\
                    for year in range(start_year, end_year)]
    leap_int = np.multiply(leap_boolean,1) # converts True/False to 1/0
    
    noleap_dpy = np.array(dpm[ds_calendar]).sum()
    leap_dpy = noleap_dpy + leap_int  
    return leap_dpy

def find_model_calendars(model):
    calendar_file = '../Processed_data/Calendars_csv/' + model + '_calendars.csv'
    #calendar_file = '../Processed_data/Calendars/' + model + '_calendars.txt'

    #model_calendars = pd.read_table(calendar_file,index_col=0, sep = ' ')
    model_calendars = pd.read_table(calendar_file,index_col=0, sep = ',')
    return model_calendars

def find_member_calendar(model, exp, member):
    model_calendars_df = find_model_calendars(model)
    exp_calendars_df = model_calendars_df[model_calendars_df['exp'] == exp]
    member_calendar_df = exp_calendars_df[exp_calendars_df['member'] == member]
    member_calendar = member_calendar_df['calendar'].values[0]
    return member_calendar
    
def plot_all_absolute_values(exp_data, piControl_linfit, corr_piControl_years, piControl_data,\
                             var_list = ['tas', 'rlut', 'rsut', 'rsdt'], text_str = None):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [18,10])
    axes = np.reshape(ax, 4)
    if text_str is not None:
        axes[0].text(0.01, 0.8, text_str, transform=axes[0].transAxes, fontsize = 14)
    years = corr_piControl_years
    for (i,var) in enumerate(var_list):
        axes[i].plot(piControl_data[var]) # piControl data
        axes[i].plot(corr_piControl_years, piControl_linfit[var].values, '--') # piControl trend
        axes[i].plot(corr_piControl_years, exp_data[var].values)
        axes[i].set_title(var, fontsize = 16)
        axes[i].tick_params(axis='both',labelsize=14)
    
def plot_anomalies(anomalies, var_list = ['tas', 'rlut', 'rsut', 'rsdt']):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [18,10])
    axes = np.reshape(ax, 4)
    #years = anomalies.index.values
    for (i,var) in enumerate(var_list):
        #axes[i].plot(years, anomalies[var])
        axes[i].plot(anomalies[var])
        axes[i].set_title(var, fontsize = 16)
        axes[i].tick_params(axis='both',labelsize=14)
        
def save_anomalies(anomalies, model, exp, member):
    filename = model + '_' + exp + '_' + member + '_anomalies.csv'
    filepath = os.path.join('../Processed_data/Global_annual_anomalies_csv/', model, exp)
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    anomalies.to_csv(filepath + '/' + filename)
    print('Anomalies for', model, exp, member, 'have been saved')
    
def piControl_timeunit_correction(model, exp, member, piControl_timeunit_start_year, piControl_start_year):
    #if model in ['FIO-ESM-2-0']:
    #    print('For', model, exp, member, 'the time unit seems reasonable',\
    #          '\n', 'and is therefore not corrected')
    #if model in []:
    #    piControl_timeunit_start_year = piControl_start_year
    
    return piControl_timeunit_start_year
    
    
#### this function needs further development: ####
def branch_time_correction(model, exp, member, branch_time_days, piControl_timeunit_start_year, piControl_start_year, years_since_piControl_start):
    initial_years_since_piControl_start = years_since_piControl_start
    
    # models with branch time given in years instead of days
    if model in ['BCC-CSM2-MR', 'BCC-ESM1']:
        if exp in hist_exp or exp in ssp_exp:
            years_since_piControl_start = int(branch_time_days - piControl_start_year)
            print('Probably branched in year', int(branch_time_days))
    elif model in ['CAMS-CSM1-0']: 
        # branch times given in years instead of days for all experiments
        years_since_piControl_start = int(branch_time_days - piControl_start_year)
        print('Probably branched in year', int(branch_time_days))
    elif model in ['CAS-ESM2-0']: # timeunit seems wrong. 
        #Should likely be years since days since 0001-01-01, when piControl starts
        years_since_piControl_start = 0 
    elif model in ['CIESM']:
        if exp in idealised_exp:
            years_since_piControl_start = 0 
            # From CIESM paper (https://doi.org/10.1029/2019MS002036): 
            # 1pctCO2, abrupt-4xCO2, historical r1 branch at the same time, jan 1st, year 1.
        if exp in hist_exp or exp in ssp_exp:
            if member == 'r1i1p1f1':
                years_since_piControl_start = 0
            else:
                years_since_piControl_start = (int(member[1])-1)*100 - 1
            # histrical r2 branch in year 100, and r3 in year 200.
            # at least the model branch info is correct that the difference is 100 years between each historical member
    elif model in ['FIO-ESM-2-0']:
        if exp in hist_exp or exp in ssp_exp:
            # "three realizations of historical simulation are conducted
            # from 1 January of year 301, 330, and 350 of the PiControl run, respectively"
            # From paper: https://doi.org/10.1029/2019JC016036
            if member == 'r1i1p1f1':
                years_since_piControl_start = 0
    elif model in ['FGOALS-f3-L']:
        if exp in idealised_exp or exp in hist_exp or exp in ssp_exp:
            piControl_branchyear = 600 + (int(member[1])-1)*50
            # According to their papers:
            # https://doi.org/10.1080/16742834.2020.1778419
            # https://doi.org/10.1007/s00376-020-2004-4
            #r1: should start in piControl year 600
            #r2: should start in piControl year 650
            #r3: should start in piControl year 700
            years_since_piControl_start = piControl_branchyear - piControl_start_year
    elif model in ['NorESM2-LM']:
        if exp in idealised_exp:
            years_since_piControl_start = 0 # because 4xCO2, 1pctCO2 branch time is the same as piControl branch time
        elif exp in hist_exp or exp in ssp_exp: # measure branch time relative to r1 member
            if member == 'r1i1p1f1':
                years_since_piControl_start = 0
            else:
                member_calendar = find_member_calendar(model, exp, member)
                days_table = np.append([0],np.cumsum(dpy(piControl_start_year,piControl_start_year+500, member_calendar)))
                branch_time_days = branch_time_days - 430335
                # find index of element closest to branch_time_days
                years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
                #print('days difference after correction:', days_table[years_since_piControl_start] - branch_time_days)
    elif model in ['NorESM2-MM']:
        years_since_piControl_start = 0 # because 4xCO2 branch time is the same as piControl branch time
    elif model in ['UKESM1-0-LL']:
        #piControl_start_year = 1960
        #if exp in expgroup1:
        #    branch_time_days = branch_time_days - 39600   
        #    years_since_piControl_start = 0
        if exp in idealised_exp or exp in hist_exp or exp in ssp_exp:
            member_calendar = find_member_calendar(model, exp, member)
            days_table = np.append([0],np.cumsum(dpy(piControl_start_year,piControl_start_year+1000, member_calendar)))
            branch_time_days = branch_time_days - 39600
            # find index of element closest to branch_time_days
            years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
            print(member)
            print(years_since_piControl_start)
            print('days difference:', days_table[years_since_piControl_start] - branch_time_days)
    elif model in ['GFDL-CM4']:
        piControl_start_year = 151 # found from manual check. Branch info is clearly wrong, since it lists different time units for piControl
        if exp == 'abrupt-4xCO2':
            years_since_piControl_start = 0
        elif exp in hist_exp or exp in ssp_exp:
            years_since_piControl_start = 100 # probably?   
    elif model == 'CanESM5': 
        if exp in idealised_exp:
            if exp == '1pctCO2':
                if int(member[-3]) == 1:
                    branch_time_days = branch_time_days - 1223115 
                elif int(member[-3]) == 2:
                    branch_time_days = branch_time_days - 1350500
                years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
                print(years_since_piControl_start)
                print('days difference:', days_table[years_since_piControl_start] - branch_time_days)
            else:
                years_since_piControl_start = 0 # because 4xCO2 branch time is the same as piControl branch time
        elif exp in hist_exp or exp in ssp_exp: # measure branch time relative to r1 member
            #piControl_start_year = 5201
            member_calendar = find_member_calendar(model, exp, member)
            days_table = np.append([0],np.cumsum(dpy(piControl_start_year,piControl_start_year+2000, member_calendar)))  
            if int(member[-3]) == 1:
                branch_time_days = branch_time_days - 1223115 
            elif int(member[-3]) == 2:
                branch_time_days = branch_time_days - 1350500
            # find index of element closest to branch_time_days
            years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
            print(member)
            print(years_since_piControl_start)
            print('days difference:', days_table[years_since_piControl_start] - branch_time_days)
    elif model in ['CanESM5-CanOE']:
        if exp in idealised_exp:
            years_since_piControl_start = 0 # because 4xCO2 branch time is the same as piControl branch time
        #elif exp == 'historical':
        elif exp in hist_exp or exp in ssp_exp:
            #piControl_start_year = 5550
            member_calendar = find_member_calendar(model, exp, member)
            days_table = np.append([0],np.cumsum(dpy(piControl_start_year,piControl_start_year+2000, member_calendar)))  
            branch_time_days = branch_time_days - 1350500
            print('branch_time_days', branch_time_days)
            # find index of element closest to branch_time_days
            years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
            print('years_since_piControl_start', years_since_piControl_start)
    elif model in ['TaiESM1']:
        if exp == '1pctCO2':
            years_since_piControl_start = 701 - 201
        elif exp == 'abrupt-4xCO2':
            years_since_piControl_start = 701 - 201 # if we choose to trust info from branch method rather than branch_time_in_parent. This info seems to be the most correct for other experiments
        elif exp == 'historical':
            years_since_piControl_start = 671 - 201
    elif model in ['EC-Earth3-Veg-LR']:
        if exp in hist_exp or exp in ssp_exp:
            branch_time_days = branch_time_days - 164359
            years_since_piControl_start = (np.abs(days_table - branch_time_days)).argmin()
            print('years_since_piControl_start', years_since_piControl_start)
    #elif model in ['KIOST-ESM'] and exp in ['historical', 'abrupt-4xCO2', '1pctCO2']:
        # piControl units: days from 1850
        #piControl_start_year = 3189 # of new version, old version in 2689
        #startyear of abrupt-4xCO2, historical, 1pctCO2 seems to be 3181, hence:
        #years_since_piControl_start = -8
        # we should include the other time period of piControl before computing anomalies
        #print('years_since_piControl_start', years_since_piControl_start)
    # KACE-1-0-G historical all members: Probably branch around year 150 in piControl instead of year 0

    
    if initial_years_since_piControl_start != years_since_piControl_start:
        print('years_since_piControl_start has been corrected to', years_since_piControl_start)
    #else:
    #    print('years_since_piControl_start has not been corrected')
    return years_since_piControl_start


