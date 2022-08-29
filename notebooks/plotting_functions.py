#plotting

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import pandas as pd
import os
from estimation import *


def plot_allvar(model, exp, members):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [18,10]); axes = np.reshape(ax, 4)
    fig.suptitle(model + ' variables', fontsize = 16)
    for member in members:
        filename = model + '_' + exp + '_' + member + '_anomalies.csv'
        file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
        data = pd.read_csv(file, index_col=0)
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
    if exp == 'abrupt-0p5xCO2':
        sign = -1
    else: 
        sign = 1
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = stopyear)
        deltaT0 = np.concatenate([[0],data[var]])
        ax.plot(years0[:len(deltaT0)], sign*deltaT0, color = color, zorder = zorder) 
    if include_mean == True:
        meanT0 = mean_4xCO2tas(model, members, length_restriction = stopyear)
        ax.plot(years0, sign*meanT0, color = color, linewidth = 3, zorder = zorder + 1);
    return ax

def plot_Nvstime(ax, model, exp, members, stopyear = 150, include_mean = True, zorder = 2):
    # just for plotting on the axis "ax", which is defined somewhere else
    
    years = np.arange(1, stopyear + 1)
    if exp == 'abrupt-0p5xCO2':
        sign = -1
    else: 
        sign = 1
            
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = stopyear)
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        ax.plot(years[:len(toarad)], sign*toarad, color = 'black', zorder = zorder)
    if include_mean == True:
        meanN = mean_4xCO2toarad(model, members, length_restriction = stopyear)
        ax.plot(years, sign*meanN, color='black', linewidth = 3, zorder = zorder + 1);
    return ax    

def plot_NvsT(ax, model, exp, members, stopyear = 150, zorder = 2):
    if exp == 'abrupt-0p5xCO2':
        sign = -1
    else: 
        sign = 1
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = stopyear)
        tas = data['tas']
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        ax.scatter(sign*tas, sign*toarad, color = 'black', zorder = zorder)
    return ax
    
def plot_NvsTfit(ax, timescales, a_n, b_n, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    tas_pred = tas_predictors(t = years0, fixed_par = timescales)
    toarad_pred = toarad_predictors(t = years0, fixed_par = timescales)
    
    tas_sum = tas_pred@a_n # sum of all components
    toarad_sum = toarad_pred@b_n # sum of all components
    
    # add extra point in equilibrium:
    tas_sum_ext = np.append(tas_sum, np.sum(a_n))
    toarad_sum_ext = np.append(toarad_sum, 0)
    ax.plot(tas_sum_ext, toarad_sum_ext, color = color, zorder = zorder)
    return ax

def plot_Tfit(ax, timescales, a_n, include_components = False, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    tas_pred = tas_predictors(t = years0, fixed_par = timescales)
    tas_sum = tas_pred@a_n # sum of all components
    ax.plot(years0, tas_sum, color = color, zorder = zorder)
    if include_components == True:
        for comp in range(len(timescales)):
            ax.plot(years0, a_n[comp]*tas_pred[:,comp], '--', linewidth=1, color = color, zorder = zorder)
    return ax

def plot_Nfit(ax, timescales, b_n, include_components = False, color = 'lightblue', linewidth=1, zorder = 1):
    years0 = np.arange(0,151)
    toarad_pred = toarad_predictors(t = years0, fixed_par = timescales)
    toarad_sum = toarad_pred@b_n # sum of all components
    ax.plot(years0, toarad_sum, color = color, zorder = zorder)
    if include_components == True:
        for comp in range(len(timescales)):
            ax.plot(years0, b_n[comp]*toarad_pred[:,comp], '--', linewidth=1, color = color, zorder = zorder)
    return ax

def fixedSSTestimate4xCO2(model, members, plotting_axis = None):
    # check if there exists a fixed-SST estimate of the forcing
    fixedSSTforcingfile = '../Estimates/fixed_SST_forcing_estimates.csv'
    fSSTf_df = pd.read_csv(fixedSSTforcingfile, index_col=0)
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
                    plotting_axis.arrow(-0.8, forcingestimate, 0.8, 0, color = "red", width = 0.15, head_length = 0.3,\
                                        clip_on = False, head_width = 0.3, length_includes_head = True)
                    #plotting_axis.arrow(-0.5, forcingestimate, 0.3, 0, color = "red",\
                    #                    clip_on = False, head_width = 0.15)


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
    ax.legend(loc=2, prop={'size': 18})

def forcing_response_figure(t, Fiarray, Tiarray, Tcoupled, titlestr = '', F4x_ratio = None, Ncoupled = None):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize = [16,7])
    for axis in ax:
        axis.set_xlabel('t (years)',fontsize = 18)
        axis.set_xlim(min(t),max(t))
        axis.grid()
        axis.tick_params(axis='both',labelsize=18)
    for i in range(0,np.shape(Fiarray)[1]-1):
        ax[0,].plot(t,Fiarray[:,i],linewidth=1,color = "gray")
    #ax[0,].plot(t,Fiarray[:,0],linewidth=2,color = "black",label = "Old forcing")
    #ax[0,].plot(t,Fiarray[:,np.shape(Fiarray)[1]-1],linewidth=1,color = "blue",label = "New forcing")
    ax[0,].plot(t,Fiarray[:,0],linewidth=2,color = "lightblue",label = "Old forcing")
    ax[0,].plot(t,Fiarray[:,np.shape(Fiarray)[1]-1],linewidth=1,color = "darkblue",label = "New forcing")
    #if Ncoupled is not None:
    #    ax[0,].plot(t, Ncoupled,linewidth=2,color = "red",label = "N")
    ax[0,].set_ylabel('F(t) [$W/m^2$]',fontsize = 18)
    ax[0,].set_title(titlestr + ' ERF',fontsize = 18)
    ax[1,].set_title(titlestr + ' Temp.',fontsize = 18)
    ax[1,].plot(t,Tcoupled,linewidth=3,color = "black",label = "coupled model response")
    #ax[1,].plot(t,Tiarray[:,0],'--',linewidth=2,color = "black",label = "Linear response to old forcing")
    ax[1,].plot(t,Tiarray[:,0],'--',linewidth=2,color = "lightblue",label = "Linear response to old forcing")
    if F4x_ratio is not None:
        ax[1,].plot(t,Tiarray[:,0]*F4x_ratio,'--',linewidth=2,color = "red",label = "Scaled response to old forcing")
    #ax[1,].plot(t,Tiarray[:,-1],'--',linewidth=2,color = "blue",label = "Linear response to new forcing")
    ax[1,].plot(t,Tiarray[:,-1],'--',linewidth=2,color = "darkblue",label = "Linear response to new forcing")
    ax[1,].set_ylabel('T(t) [°C]',fontsize = 18)
    #print(np.size(Tcoupled))
    #Tstd = np.std(expfit_detrend(Tcoupled.values))
    #inds = np.argwhere(np.abs(Tcoupled.values-Tiarray[:,-1])/Tstd>2)#[:,-1]
    #print(len(inds))
    #if len(inds)>0:
    #    ax[1,].axvline(x=t[inds[0]])
    ax[0,].text(0,1.03,'a)',transform=ax[0,].transAxes, fontsize=20)
    ax[1,].text(0,1.03,'b)',transform=ax[1,].transAxes, fontsize=20)
    #plt.close()
    return fig, ax

