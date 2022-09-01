# estimation

import random as rd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import os
from scipy import optimize
# causes problems with circular referencing:
#from plotting_functions import *


def random_tau(dim=3): # return "dim" randomly chosen time scales
    # the log10 of the time scale will be the random number:
    tau1 = 10**rd.uniform(np.log10(1), np.log10(6)) # range: (1, 6) years
    factor1 = rd.uniform(5,10); tau2 = factor1*tau1;
    tau3 = 10**rd.uniform(np.log10(80), np.log10(1000))
    
    # fixed time scales:
    taulist = np.array([tau1, tau2, tau3])
    return taulist

def find_members(model, exp, datatype = 'anomalies'):
    if datatype == 'anomalies':
        directory = '../Processed_data/Global_annual_anomalies/'
    elif datatype == 'means':
        directory = '../Processed_data/Global_annual_means/'
    elif datatype == 'forcing':
        directory = '../Estimates/Transient_forcing_estimates/'
    modelexpdirectory = os.path.join(directory, model, exp)
    filenames = [f.name for f in os.scandir(modelexpdirectory) if f.name not in ['.ipynb_checkpoints', '.DS_Store']]

    members = [file.rsplit('_')[2] for file in filenames]
    members.sort()
    return members

def load_anom(model, exp, member, length_restriction = None):
    filename = model + '_' + exp + '_' + member + '_anomalies.csv'
    file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
    
    data = pd.read_csv(file, index_col=0)
    if model != 'AWI-CM-1-1-MR': # maybe not neccessary after updating the data archive?
        data = data.dropna() #.reset_index()
    if length_restriction != None:
        data = data[:length_restriction]
    return data

# this function has a new name that needs to be updated in scripts:
def member_mean_tas(model, members, length_restriction = None, exp = 'abrupt-4xCO2'):
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
    # compute components/predictors for T_n(t) = exp(-t/tau_n)*F(t) (* is a convolution)
    # input for stepforcing: years, fixed parameters (timescales for stepforcing)
    # stepforcing_ computes response to unit forcing,
    # to be multiplied by the actual forcing afterwards
    
    # timevaryingforcing: need a forcing time series input
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
            data = load_anom(model, exp, member, length_restriction = stopyear)
            #if model != 'AWI-CM-1-1-MR':
            #    # maybe not neccessary to dropna after updating the data archive?
            #    data = data.dropna().reset_index()
            #display(data)
            tas = data['tas']; #deltaT0 = np.concatenate([[0], tas])
            toarad = data['rsdt'] - data['rsut'] - data['rlut']
            if exp == 'abrupt-0p5xCO2':
                tas = -tas; toarad = -toarad # change signs before estimation
            tas_dict[member] = tas; toarad_dict[member] = toarad

        meanT0 = member_mean_tas(model, members, length_restriction = stopyear, exp = exp)
        if exp == 'abrupt-0p5xCO2':
            meanT0 = -meanT0
        dim = 3 # the number of time scales
        if any(np.isnan(fixed_timescales)):
            taulist = random_tau(dim)
        else:
            taulist = fixed_timescales
        # compute predictors (1 - np.exp((-t/tau[i])) of the linear model
        tas_pred = tas_predictors(t = years0, fixed_par = taulist);
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
            reg_endindex3 = endindex(len(tas), len(T_region3), taulist[2])
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
        T4 = a_4*tas_predictors(t = years0, fixed_par = [tau4])[:,0]
        #N4 = b_4*toarad_predictors(t = years0, fixed_par = tau4)
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
            reg_endindex2 = endindex(len(tas), len(T_region2), taulist[1])
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
            reg_endindex1 = endindex(len(tas), len(T_region1), taulist[0])
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
            #fig = Gregory_plot()
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            # x-coordinate: data coordinates, y-coordinate on axis coordinates (0,1)
            texts = ['$a_1$', '$a_1 + a_2$', '$a_1 + a_2 + a_3$']
            for (l,splitline) in enumerate(splits):
                ax.axvline(x=splitline)
                ax.text(splitline-0.1, 0.95, texts[l], fontsize = 14,\
                        horizontalalignment='right', transform = trans)
            for member in members:
                data = load_anom(model,exp, member, length_restriction = 150)
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

def Gregory_linreg(model, exp, members, startyear = 1, stopyear = 150):
    alltas = []; alltoarad = []
    # use datapoints from all specified members in the regression
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = stopyear)
        data = data[(startyear-1):]
        tas = data['tas']
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        if exp == 'abrupt-0p5xCO2': # change sign of data
            tas = -tas; toarad = -toarad
        alltas = np.append(alltas, tas)
        alltoarad = np.append(alltoarad,toarad)
        
    regpar = np.polyfit(alltas, alltoarad, 1)
    if exp == 'abrupt-4xCO2':
        gF2x = regpar[1]/2; gT2x = -regpar[1]/(2*regpar[0])
        linfit = np.polyval(regpar, [0, gT2x*2])
    else:
        gF2x = regpar[1]; gT2x = -regpar[1]/(regpar[0])
        linfit = np.polyval(regpar, [0, gT2x])
    return gF2x, gT2x, linfit

def forcing_F13(tasdata, Ndata, model, inputparfile = 'best_estimated_parameters_allmembers4xCO2.csv', years = None):
    parameter_table = pd.read_csv('../Estimates/' + inputparfile,index_col=0)
    if years == None:
        GregoryT2x = parameter_table.loc[model,'GregoryT2x']
        GregoryF2x = parameter_table.loc[model,'GregoryF2x']
    if years == '1-20':
        GregoryT2x = parameter_table.loc[model,'GregoryT2x_1-20']
        GregoryF2x = parameter_table.loc[model,'GregoryF2x_1-20']
    fbpar = GregoryF2x/GregoryT2x
    F = Ndata + fbpar*tasdata
    return F

def etminan_co2forcing(C, C0 = 284.3169998547858, N_bar = 273.021049007482):
    a1 = -2.4*10**(-7)
    b1 = 7.2*10**(-4)
    c1 = -2.1*10**(-4)
    return (a1*(C-C0)**2 + b1*np.abs(C-C0) + c1*N_bar + 5.36)*np.log(C/C0)

def expfit_detrend(x):
    dim = 3
    taulist = random_tau(dim)
    tas_pred = tas_predictors(t = np.arange(0,150), fixed_par = taulist)
    a_n, rnorm1 = optimize.nnls(tas_pred, np.abs(x))
    Ti = np.array([tas_pred[:,i]*a_n[i] for i in range(0,dim)]) # compute components
    Tsum = tas_pred@a_n # sum of all components
    return (np.abs(x) - Tsum)*np.sign(x[-1])

def recT_from4xCO2(T4xCO2, F4xCO2, forcing_rec_exp):
    # Use what we know about 4xCO2 forcing and temperature response T4xCO2
    # to reconstruct the temperature response to the forcing "forcing_rec_exp"
    # index 0 should correspond to year 0, where T4xCO2[0]=0,
    # and we set the forcing difference in year 0 to 0
    lf = len(forcing_rec_exp)
    delta_f_vector = np.concatenate(([0], np.diff(forcing_rec_exp)))
    
    if lf > len(T4xCO2):
        print('Cannot compute reconstruction for an experiment longer than the 4xCO2 experiment')
    else:
        W = np.full((lf,lf),0.) # will become a lower triangular matrix
        for i in range(0,lf):
            for j in range(0,i):
                W[i,j] = delta_f_vector[i-j]/F4xCO2
        #print(W)
        T_rec_exp = W@(T4xCO2[:lf])
        return T_rec_exp
        #return W