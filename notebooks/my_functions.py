"""Define functions that can be used in multiple notebooks"""

import random as rd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from scipy import signal
import pandas as pd
import os
import my_functions as f
from scipy import optimize


def area_weights(lat_bnds, lon_bnds): 
    # computes exact area weigths assuming earth is a perfect sphere
    lowerlats = np.radians(lat_bnds[:,0]); upperlats = np.radians(lat_bnds[:,1])
    difflon = np.radians(np.diff(lon_bnds[0,:])) 
    # if the differences in longitudes are all the same
    areaweights = difflon*(np.sin(upperlats) - np.sin(lowerlats));
    areaweights /= areaweights.mean()
    return areaweights # list of weights, of same dimension as latitude

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
    else: 
        norm_by_annual = np.array([month_length[i*12:(i+1)*12].mean() for i in range(int(len(month_length)/12))])
        norm_by_monthly = np.concatenate([np.tile(norm_by_annual[i], 12) for i in range(len(norm_by_annual))])
        
    weights = month_length/norm_by_monthly
    # normalized to have mean 1
    return weights

# days per year
def dpy(model, start_year, end_year):
    ds_calendar = f.calendar_check(model) 
    leap_boolean = [f.leap_year(year, calendar = ds_calendar)\
                    for year in range(start_year, end_year)]
    leap_int = np.multiply(leap_boolean,1)
    
    noleap_dpy = np.array(dpm[ds_calendar]).sum()
    leap_dpy = noleap_dpy + leap_int
    return leap_dpy  


## define functions
def calendar_check(model):
    calendarfile = '../Processed_data/Calendars/' + model + '_calendars.txt'
    cal_df = pd.read_table(calendarfile, index_col=0, sep = ' ')
    calendars_used = cal_df['calendar'].drop_duplicates()
    # I think 365 days calendar must be the same as noleap
    return calendars_used.values[0]



def random_tau(dim=3): # return "dim" randomly chosen time scales
    # the log10 of the time scale will be the random number:
    tau1 = 10**rd.uniform(np.log10(1), np.log10(6)) # range: (1, 6) years
    factor1 = rd.uniform(5,10); tau2 = factor1*tau1;
    tau3 = 10**rd.uniform(np.log10(80), np.log10(1000))
    
    # fixed time scales:
    taulist = np.array([tau1, tau2, tau3])
    return taulist

def plot_allvar(model, exp, members):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [18,10]); axes = np.reshape(ax, 4)
    fig.suptitle(model + ' variables', fontsize = 16)
    for member in members:
        filename = model + '_' + exp + '_' + member + '_anomalies.txt'
        file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
        data = pd.read_table(file, index_col=0, sep = ',')
        data = data.dropna().reset_index()
        x = data.index.values

        # plot each variable:
        for (i,var) in enumerate(['tas', 'rlut', 'rsut', 'rsdt']):
            axes[i].plot(x, data[var])
            axes[i].set_title(var, fontsize = 16)
            axes[i].tick_params(axis='both',labelsize=14)
    plt.show()
    #return ax
    
def plot_allvar_preloaded(xvalues, dataframe, model, member, plot_title = ' '):
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize = [16,10])
    fig.suptitle(plot_title,fontsize = 18)
    ax = np.concatenate(axes)
    variables = list(dataframe)
    if 'index' in variables:
        variables.remove('index')
    units = [' [K]', ' [W/$m^2$]', ' [W/$m^2$]', ' [W/$m^2$]']
    for (j, var) in enumerate(variables):
        ax[j].plot(xvalues, dataframe[var], linewidth=2,color = "black")
        ax[j].set_ylabel(var + ' ' + units[j],fontsize = 18)
        ax[j].set_xlabel('Year',fontsize = 18)
        ax[j].tick_params(axis='both',labelsize=18)
        ax[j].set_xlim(min(xvalues),max(xvalues))
        ax[j].grid()
        
def plot_tasandN(xvalues, tasdata, Ndata, plot_title = ''):
    # plot just tas and netTOA radiative imbalance
    fig, ax = plt.subplots(ncols=2,figsize = [16,5])
    fig.suptitle(plot_title, fontsize = 18)
    ax[0].plot(xvalues, tasdata, linewidth=2, color = "black"); ax[0].set_ylabel('tas [K]',fontsize = 18)
    ax[1].plot(xvalues, Ndata, linewidth=2, color = "black");  ax[1].set_ylabel('N [W/$m^2$]',fontsize = 18)
    for j in range(len(ax)):
        ax[j].set_xlabel('Year',fontsize = 18)
        ax[j].tick_params(axis='both',labelsize=18)
        ax[j].set_xlim(min(xvalues),max(xvalues))
        ax[j].grid()

def find_members(model, exp, datatype = 'anomalies'):
    if datatype == 'anomalies':
        directory = '../Processed_data/Global_annual_anomalies/'
    elif datatype == 'means':
        directory = '../Processed_data/Global_annual_means/'
    modelexpdirectory = os.path.join(directory, model, exp)
    filenames = [f.name for f in os.scandir(modelexpdirectory) if f.name !='.ipynb_checkpoints']

    members = [file.rsplit('_')[2] for file in filenames]
    members.sort()
    return members

def load_anom(model, exp, member, length_restriction = None):
    filename = model + '_' + exp + '_' + member + '_anomalies.txt'
    file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
    data = pd.read_table(file, index_col=0, sep = ',')
    data = data.dropna().reset_index()
    if length_restriction != None:
        data = data[:length_restriction]
    return data

def mean_4xCO2tas(model, members, length_restriction = None, exp = 'abrupt-4xCO2'):
    # add also 0 response at time 0
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = length_restriction)
        deltaT0 = np.concatenate([[0],data['tas']])
        if mb == 0:
            if length_restriction == None:
                length_restriction = len(deltaT0)-1 # full length of data
            T_array = np.full((len(members), length_restriction + 1), np.nan) 
        T_array[mb,:len(deltaT0)] = deltaT0
    meanT0 = np.nanmean(T_array, axis=0)
    return meanT0

def mean_4xCO2toarad(model, members, length_restriction = None, exp = 'abrupt-4xCO2'):
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = length_restriction)
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        if mb == 0:
            if length_restriction == None:
                length_restriction = len(toarad) # full length of data
            N_array = np.full((len(members), length_restriction), np.nan) 
        N_array[mb,:len(toarad)] = toarad
    meanN = np.nanmean(N_array, axis=0)
    return meanN

def tas_predictors(t, fixed_par, exptype = 'stepforcing', timevaryingforcing = []):
    # input for stepforcing: years, fixed parameters (timescales for stepforcing)
    # compute components/predictors for T_n(t) = exp(-t/tau_n)*F(t) (* is a convolution)
    if exptype == 'stepforcing':
        timescales = fixed_par; dim = len(timescales)
        predictors = np.zeros((len(t),dim))
        for i in range(0,dim): 
            predictors[:,i] = (1 - np.exp((-t)/timescales[i]))
    elif exptype == 'timevaryingforcing': # need forcing input
        # compute components T_n(t) = exp(-t/tau_n)*F(t) (Here * is a convolution)
        timescales = fixed_par
        lf = len(timevaryingforcing); dim = len(timescales)
        predictors = np.full((lf,dim),np.nan)   

        # compute exact predictors by integrating greens function
        for k in range(0,dim):
            # dot after 0 to create floating point numbers:
            intgreensti = np.full((lf,lf),0.)   
            for t in range(0,lf):
                # compute one new contribution to the matrix:
                intgreensti[t,0] = timescales[k]*(np.exp(-t/timescales[k])\
                                               - np.exp(-(t+1)/timescales[k]))
                # take the rest from row above:
                if t > 0:
                    intgreensti[t,1:(t+1)] = intgreensti[t-1,0:t]
            # compute discretized convolution integral by this matrix product:
            predictors[:,k] = intgreensti@np.array(timevaryingforcing)
    else:
        print('unknown exptype')
    return predictors

def toarad_predictors(t, fixed_par, exptype = 'stepforcing'):
    # input: years, fixed parameters (timescales for stepforcing)
    # compute components/predictors for N_n(t)
    if exptype == 'stepforcing':
        timescales = fixed_par
        dim = len(timescales)
        A = np.zeros((len(t),dim))
        for i in range(0,dim): 
            A[:,i] = np.exp((-t)/timescales[i])
    else:
        print('Function needs further development')
    return A
        
def Gregory_plot(fontsize = 18, x_max = None, y_max = None, figsize = [8,6]):
    fig = plt.figure(figsize = figsize);
    plt.xlabel('T [°C]',fontsize = fontsize)
    plt.ylabel('N [$W/m^2$]',fontsize = fontsize)
    plt.axhline(0, color='k', linewidth = 0.5) # horizontal lines
    plt.tick_params(axis='both',labelsize=18)  
    if x_max != None:
        plt.set_xlim(0,x_max)
    if y_max != None:
        plt.set_ylim(0,y_max)
    plt.close()
    return fig

def endindex(lenT, lenTcomp, timescale, p_lim = 0.01):
    # lenTcomp: length of component of interest
    # lenT: length of full time series
    p_lim_index = np.int(np.ceil(-timescale*np.log(p_lim)))-1
    # based on the full time series. lenTcomp could be smaller than this, and then we use all points lenT.
    # Otherwise, we need to subtract the number of points of the
    # full time series not used in the component of interest
    reg_endindex = np.min([p_lim_index - (lenT-lenTcomp), lenTcomp]); 
    return reg_endindex

def fbpar_estimation(model, exp, members, plotting_axis = None, stopyear = 150, p_lim = 0.01, fixed_timescales = np.full(3, np.nan)):
    # if not inputting values for fixed_timescales, they will be randomly chosen
    # if we want to plot, use option: plotting_axis = ax
    try:
        #### load data and estimate parameters of temperature response ####
        years = np.arange(1, stopyear + 1); years0 = np.arange(0, stopyear + 1)
        tas_dict = {}; toarad_dict = {}
        for (mb, member) in enumerate(members):
            data = f.load_anom(model, exp, member, length_restriction = stopyear)
            tas = data['tas']; #deltaT0 = np.concatenate([[0], tas])
            toarad = data['rsdt'] - data['rsut'] - data['rlut']
            tas_dict[member] = tas; toarad_dict[member] = toarad

        meanT0 = f.mean_4xCO2tas(model, members, length_restriction = stopyear)

        dim = 3 # the number of time scales
        if any(np.isnan(fixed_timescales)):
            taulist = f.random_tau(dim)
        else:
            taulist = fixed_timescales
        # compute predictors (1 - np.exp((-t/tau[i])) of the linear model
        tas_pred = f.tas_predictors(t = years0, fixed_par = taulist);
        # find parameters a_n in: deltaT = \sum_i a_n[i]*(1 - np.exp((-t/tau[i]))
        a_n, rnorm1 = optimize.nnls(tas_pred,meanT0) # non-negative least squares, to ensure positive parameters
        Ti = np.array([tas_pred[:,i]*a_n[i] for i in range(0,dim)]) # compute components
        Tsum = tas_pred@a_n # sum of all components
        splits = np.cumsum(a_n);


        #### estimate feedback parameters ####
        T_region3_reg = []; N_region3_reg = []; 
        #T_region3_all = []; N_region3_all = []
        for member in members:
            tas = tas_dict[member]; toarad = toarad_dict[member]
            T_region3 = tas[tas>=splits[-2]]; N_region3 = toarad[tas>splits[-2]]
            #T_region3_all = np.append(T_region3_all, T_region3)
            #N_region3_all = np.append(N_region3_all, N_region3)
            reg_endindex3 = f.endindex(len(tas), len(T_region3), taulist[2])
            T_region3_reg = np.append(T_region3_reg, T_region3[:reg_endindex3])
            N_region3_reg = np.append(N_region3_reg, N_region3[:reg_endindex3])    
        if len(T_region3_reg) < 10:
            raise Exception('Too few points for regression')
        reg3par = np.polyfit(T_region3_reg, N_region3_reg, deg = 1)
        b_4 = reg3par[1] + reg3par[0]*np.sum(a_n)
        lambda3 = - reg3par[0]; lambda4 = lambda3; a_4 = b_4/lambda4
        #T_reg3_ends = np.concatenate([[0], [a_4+sum(a_n)]])
        T_reg3_ends = np.concatenate([[splits[1]], [a_4+sum(a_n)]])
        linfit3 = np.polyval(reg3par, T_reg3_ends)    # for plotting

        # need to include the "invisible" even longer time scale for this to work:
        tau4 = 5000;
        T4 = a_4*f.tas_predictors(t = years0, fixed_par = [tau4])[:,0]
        #N4 = b_4*f.toarad_predictors(t = years0, fixed_par = tau4)
        rnorm2 = np.linalg.norm(Tsum + T4 - meanT0)
        # but check that this component is approximately 0 over the first 150 years

        # subtract components associated with third and fourth time scale:
        T3 = Ti[-1,1:]; N3 = (lambda3*(a_n[-1] - T3)) # = lambda3*a_n[2]*np.exp(-years/taulist[2]) 
        T_region2_reg = []; N_region2_reg = []; T_region2_all = []; N_region2_all = []
        for member in members:
            tas = tas_dict[member]; toarad = toarad_dict[member]
            T1and2 = tas - T3[:len(tas)]; N1and2 = toarad - N3[:len(toarad)] - b_4
            T_region2 = T1and2[T1and2>=splits[0]]; 
            N_region2 = N1and2[T1and2>=splits[0]]
            T_region2_all = np.append(T_region2_all, T_region2)
            N_region2_all = np.append(N_region2_all, N_region2)
            reg_endindex2 = f.endindex(len(tas), len(T_region2), taulist[1])
            if reg_endindex2 <= 2: # if region of regression is 0-2 points
                reg_endindex2 = 3 # make sure we use at least 3 points if available
            T_region2_reg = np.append(T_region2_reg, T_region2[:reg_endindex2])
            N_region2_reg = np.append(N_region2_reg, N_region2[:reg_endindex2])
        T_region2_reg_mirrored = splits[1] - T_region2_reg # make dT2 a mirror of T2, and starting in 0, just for parameter estimation
        # estimate parameters in linear model with no constant term
        ls_sol2, res, rank, sing_vals = np.linalg.lstsq(np.array([T_region2_reg_mirrored]).T, np.array([N_region2_reg]).T, rcond=None)
        T_reg2_ends = np.array([np.concatenate(([splits[0]], [splits[1]]))]).T
        linfit2 = - T_reg2_ends@ls_sol2 + splits[1]*ls_sol2[0,0]

        lambda2 = ls_sol2[0,0];
        T2 = Ti[-2,1:]; N2 = (lambda2*(a_n[-2] - T2))
        T_region1_reg = []; N_region1_reg = []; T_region1_all = []; N_region1_all = []
        for member in members:
            tas = tas_dict[member]; toarad = toarad_dict[member]
            T1and2 = tas - T3[:len(tas)]; N1and2 = toarad - N3[:len(toarad)] - b_4
            T1 = T1and2 - T2[:len(tas)]; N1 = N1and2 - N2[:len(toarad)]
            T_region1 = T1; N_region1 = N1
            T_region1_all = np.append(T_region1_all, T_region1)
            N_region1_all = np.append(N_region1_all, N_region1)
            reg_endindex1 = f.endindex(len(tas), len(T_region1), taulist[0])
            T_region1_reg = np.append(T_region1_reg, T_region1[:reg_endindex1])
            N_region1_reg = np.append(N_region1_reg, N_region1[:reg_endindex1])
        T_region1_reg_mirrored = splits[0] - T_region1_reg
        # estimate parameters in linear model with no constant term
        ls_sol1, res, rank, sing_vals = np.linalg.lstsq(np.array([T_region1_reg_mirrored]).T, np.array([N_region1_reg]).T, rcond=None)
        T_reg1_ends = np.array([np.concatenate(([0], [splits[0]]))]).T
        linfit1 = - T_reg1_ends@ls_sol1 + splits[0]*ls_sol1[0,0]


        # put all estimated lambdas in array, and estimate b_n:
        lambdas = np.array([ls_sol1[0,0], ls_sol2[0,0], - reg3par[0], - reg3par[0]])
        a_n = np.append(a_n, a_4); b_n = lambdas*a_n; taulist = np.append(taulist, 5000)
        return_list = [taulist, a_n, b_n, rnorm1, rnorm2]

        if plotting_axis != None:
            ax = plotting_axis
            #fig = f.Gregory_plot()
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            # x-coordinate: data coordinates, y-coordinate on axis coordinates (0,1)
            texts = ['$a_1$', '$a_1 + a_2$', '$a_1 + a_2 + a_3$']
            for (l,splitline) in enumerate(splits):
                ax.axvline(x=splitline)
                ax.text(splitline-0.1, 0.95, texts[l], fontsize = 14,\
                        horizontalalignment='right', transform = trans)
            for member in members:
                data = f.load_anom(model,exp, member, length_restriction = 150)
                tas = data['tas']; 
                toarad = data['rsdt'] - data['rsut'] - data['rlut']
                ax.scatter(tas, toarad, color = 'black')

            ax.plot(T_reg3_ends, linfit3, '--', color = 'blue', linewidth=1)
            ax.scatter(T_region2_all, N_region2_all, color = 'gray')
            ax.plot(T_reg2_ends, linfit2, '--', color = 'blue', linewidth=1);
            ax.scatter(T_region1_all, N_region1_all, color = 'lightgray')
            ax.plot(T_reg1_ends, linfit1, '--', color = 'blue', linewidth=1);
            return return_list, ax
        else:
            return return_list
    except Exception:
        #print('Too few points for regression')
        return []
    
def Gregory_subplot(ax, fontsize = 18, x_max = None, y_max = None):
    ax.set_xlabel('T [°C]',fontsize = fontsize)
    ax.set_ylabel('N [$W/m^2$]',fontsize = fontsize)
    ax.axhline(0, color='k', linewidth = 0.5) # horizontal lines
    ax.tick_params(axis='both',labelsize=18)
    ax.set_title('Evolution of top of the atmosphere radiative imbalance and temperature', fontsize = fontsize)
    if x_max != None:
        ax.set_xlim(0,x_max)
    if y_max != None:
        ax.set_ylim(-1,y_max)
    #plt.close()
    return ax

def tasresponse_subplot(ax, fontsize = 18, x_max = 150):
    ax.set_xlabel('Years after quadrupling',fontsize = fontsize)
    ax.set_ylabel('T [°C]',fontsize = fontsize)
    ax.tick_params(axis='both',labelsize=18)
    ax.set_title('$T$ following abrupt4xCO$_2$',fontsize = 18)
    ax.set_xlim(0,x_max)
    return ax
    
def toaradresponse_subplot(ax, fontsize = 18, x_max = 150, y_max = None):
    ax.set_xlabel('Years after quadrupling',fontsize = fontsize)
    ax.set_ylabel('N [$W/m^2$]',fontsize = fontsize)
    ax.tick_params(axis='both',labelsize=18)
    ax.set_title('$N$ following abrupt4xCO$_2$',fontsize = 18)
    ax.set_xlim(0,x_max)
    if y_max != None:
        ax.set_ylim(-0.6,y_max)
    return ax

def plot_Tvstime(ax, model, exp, members, stopyear = 150, include_mean = True, zorder = 2, color = 'black'):
    # just for plotting on the axis "ax", which is defined somewhere else
    
    var = 'tas'; years0 = np.arange(0, stopyear + 1)
    for (mb, member) in enumerate(members):
        data = f.load_anom(model, exp, member, length_restriction = stopyear)
        deltaT0 = np.concatenate([[0],data[var]])
        ax.plot(years0[:len(deltaT0)], deltaT0, color = color, zorder = zorder) 
    if include_mean == True:
        meanT0 = f.mean_4xCO2tas(model, members, length_restriction = stopyear)
        ax.plot(years0, meanT0, color = color, linewidth = 3, zorder = zorder + 1);
    return ax

def plot_Nvstime(ax, model, exp, members, stopyear = 150, include_mean = True, zorder = 2):
    # just for plotting on the axis "ax", which is defined somewhere else
    
    years = np.arange(1, stopyear + 1)
    for (mb, member) in enumerate(members):
        data = f.load_anom(model, exp, member, length_restriction = stopyear)
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        ax.plot(years[:len(toarad)], toarad, color = 'black', zorder = zorder) 
    if include_mean == True:
        meanN = f.mean_4xCO2toarad(model, members, length_restriction = stopyear)
        ax.plot(years, meanN, color='black', linewidth = 3, zorder = zorder + 1);
    return ax    

def plot_NvsT(ax, model, exp, members, stopyear = 150, zorder = 2):
    for (mb, member) in enumerate(members):
        data = f.load_anom(model, exp, member, length_restriction = stopyear)
        tas = data['tas']
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        ax.scatter(tas, toarad, color = 'black', zorder = zorder)
    return ax
    
def plot_NvsTfit(ax, timescales, a_n, b_n, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    tas_pred = f.tas_predictors(t = years0, fixed_par = timescales)
    toarad_pred = f.toarad_predictors(t = years0, fixed_par = timescales)
    
    tas_sum = tas_pred@a_n # sum of all components
    toarad_sum = toarad_pred@b_n # sum of all components
    
    # add extra point in equilibrium:
    tas_sum_ext = np.append(tas_sum, np.sum(a_n))
    toarad_sum_ext = np.append(toarad_sum, 0)
    ax.plot(tas_sum_ext, toarad_sum_ext, color = color, zorder = zorder)
    return ax

def plot_Tfit(ax, timescales, a_n, include_components = False, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    tas_pred = f.tas_predictors(t = years0, fixed_par = timescales)
    tas_sum = tas_pred@a_n # sum of all components
    ax.plot(years0, tas_sum, color = color, zorder = zorder)
    if include_components == True:
        for comp in range(len(timescales)):
            ax.plot(years0, a_n[comp]*tas_pred[:,comp], '--', linewidth=1, color = color, zorder = zorder)
    return ax

def plot_Nfit(ax, timescales, b_n, include_components = False, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    toarad_pred = f.toarad_predictors(t = years0, fixed_par = timescales)
    toarad_sum = toarad_pred@b_n # sum of all components
    ax.plot(years0, toarad_sum, color = color, zorder = zorder)
    if include_components == True:
        for comp in range(len(timescales)):
            ax.plot(years0, b_n[comp]*toarad_pred[:,comp], '--', linewidth=1, color = color, zorder = zorder)
    return ax

def Gregory_linreg(model, exp, members, startyear = 1, stopyear = 150):
    alltas = []; alltoarad = []
    # use datapoints from all specified members in the regression
    for (mb, member) in enumerate(members):
        data = f.load_anom(model, exp, member, length_restriction = stopyear)
        data = data[(startyear-1):]
        tas = data['tas']
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        alltas = np.append(alltas, tas)
        alltoarad = np.append(alltoarad,toarad)
        
    regpar = np.polyfit(alltas, alltoarad, 1)
    gF2x = regpar[1]/2; gT2x = -regpar[1]/(2*regpar[0])
    linfit = np.polyval(regpar, [0, gT2x*2])
    return gF2x, gT2x, linfit

def fixedSSTestimate4xCO2(model, members, plotting_axis = None):
    # check if there exists a fixed-SST estimate of the forcing
    fixedSSTforcingfile = '../Estimates/fixed_SST_forcing_estimates.txt'
    fSSTf_df = pd.read_table(fixedSSTforcingfile, index_col=0, sep = '\t')
    nmembers = len(fSSTf_df.index[fSSTf_df.index == model])

    if nmembers>0:
        if nmembers == 1:
            model_df = pd.DataFrame([fSSTf_df.loc[model]])
        else:
            model_df = fSSTf_df.loc[model]
        memberlist = model_df['member'].values
        #print('Fixed-SST forcing estimate exists for', nmembers,\
        #      'member(s) of this model:', memberlist)
        for member in memberlist:
            fSSTrow = model_df[model_df['member'] == member]
            fSSTf = fSSTrow['F_4xCO2']
            if plotting_axis != None:
                plotting_axis.scatter(fSSTrow['$\Delta$T'].values, fSSTrow['F_4xCO2'].values, label = 'Fixed-SST forcing',\
                       color = "red", marker = 'x', s=200, linewidth = 3,zorder=1000)
                print('Fixed-SST forcing for member', member, 'is', np.round(fSSTf[0],3), 'W/m^2')

def forcing_smith(model, ERFtype = 'ERFtrop', plotting_axis = None):
    ERF_file = '../Estimates/ERF_methods_4xCO2_smith2020.csv'
    ERF_df = pd.read_csv(ERF_file, index_col=0)
    ERF_df = ERF_df.set_index(ERF_df['Model'])
    nmembers = len(ERF_df.index[ERF_df.index == model])
    if nmembers>0:
        if nmembers == 1:
            model_df = pd.DataFrame([ERF_df.loc[model]])
        else:
            model_df = ERF_df.loc[model]
        model_df = model_df.set_index(model_df['Run'])
        memberlist = model_df.index
        for member in memberlist:
            forcingestimate = model_df.loc[member][ERFtype]
            if plotting_axis != None:
                if ERFtype == 'ERFtrop':
                    plotting_axis.arrow(-0.5, forcingestimate, 0.4, 0, color = "red",\
                                        clip_on = False, head_width = 0.15)

def forcing_F13(tasdata, Ndata, model, inputparfile = 'best_estimated_parameters_allmembers.txt'):
    parameter_table = pd.read_table('../Estimates/' + inputparfile,index_col=0)
    GregoryT2x = parameter_table.loc[model,'GregoryT2x']
    GregoryF2x = parameter_table.loc[model,'GregoryF2x']
    fbpar = GregoryF2x/GregoryT2x
    F = Ndata + fbpar*tasdata
    return F
                    
                
def plot_components(t, Tn, timescales):
    fig, ax = plt.subplots(figsize = [9,5]) 
    plt.plot(t,Tn[:,0],linewidth=2,color = "black",label = 'Mode with time scale ' + str(np.round(timescales[0])) + ' years')
    plt.plot(t,Tn[:,1],linewidth=2,color = "blue",label = 'Mode with time scale ' + str(np.round(timescales[1])) + ' years')
    if len(timescales)>2:
        plt.plot(t,Tn[:,2],linewidth=2,color = "red",label = 'Mode with time scale ' + str(np.round(timescales[2],1)) + ' years')
    ax.set_xlabel('t',fontsize = 18)
    ax.set_ylabel('T(t)',fontsize = 18)
    ax.set_title('Temperature responses to forcing',fontsize = 18)
    ax.grid()
    ax.set_xlim(min(t),max(t))
    ax.tick_params(axis='both',labelsize=22)
    ax.legend(loc=2, prop={'size': 18});
    

def forcing_response_figure(t, Fiarray, Tiarray, Tcoupled, sspexp = '', model = ''):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize = [16,5])
    for axis in ax:
        axis.set_xlabel('t (years)',fontsize = 18)
        axis.set_xlim(min(t),max(t))
        axis.grid()
        axis.tick_params(axis='both',labelsize=18)
    for i in range(0,np.shape(Fiarray)[1]-1):
        ax[0,].plot(t,Fiarray[:,i],linewidth=1,color = "gray")
    ax[0,].plot(t,Fiarray[:,0],linewidth=2,color = "black",label = "Old forcing")
    ax[0,].plot(t,Fiarray[:,np.shape(Fiarray)[1]-1],linewidth=1,color = "blue",label = "New forcing")
    ax[0,].set_ylabel('F(t) [$W/m^2$]',fontsize = 18)
    if sspexp == '':
        ax[0,].set_title('Effective radiative forcing',fontsize = 18)
        ax[1,].set_title('Temperature',fontsize = 18)
    else:
        ax[0,].set_title('Historical and ' + sspexp + ' effective forcing',fontsize = 18)
        ax[1,].set_title('Historical and ' + sspexp + ' temperature',fontsize = 18)
    ax[1,].plot(t,Tcoupled,linewidth=3,color = "black",label = model + " modelled response")
    ax[1,].plot(t,Tiarray[:,0],'--',linewidth=2,color = "black",label = "Linear response to old forcing")
    ax[1,].plot(t,Tiarray[:,-1],'--',linewidth=2,color = "blue",label = "Linear response to new forcing")
    ax[1,].set_ylabel('T(t) [°C]',fontsize = 18)
    
    
    ax[0,].text(0,1.03,'a)',transform=ax[0,].transAxes, fontsize=20)
    ax[1,].text(0,1.03,'b)',transform=ax[1,].transAxes, fontsize=20)
    #plt.close()
    return fig
    





