# Code and data to be used for a paper about CMIP6 forcing

## Step 1: Find out which data are available for analysis
The work done in this step is stored in the folder Data_availability.
First we search through ESGF using pyclient to find all models with both piControl and abrupt-4xCO2 experiments.
For these models, we search through ESGF to find all available data for the experiments we are interested in.
Furthermore, we check which of these data are available through Google Cloud, and store this info for later use.

However, the availability of data change with time, so we have also included some files describing updates to the initial availability search, and problems we found with the data:
It seems some data are removed from ESGF before downloading. Info about these data are stored in Data_availability/ESGF/esgf-missing_march22nd.json. Some files are not used because they contain errors. Info about these are stored in Data_availability/ESGF/files_with_errors.json.
We note also that some files must have been added to ESGF after our initial search. Some additional files we have included are listed in the file Data_availability/ESGF/esgf-added_march22nd.json. 

All data not available in GoogleCloud are downloaded. This includes also some files that we found to be in GoogleCloud, but which contained errors: Data_availability/GoogleCloud/cloud-buterrors.json, and some files that have been removed from the cloud: Data_availability/GoogleCloud/cloud-removed_march17.json.

## Step 2: Constructing global annual means and anomalies
The code for this work is stored in the folder Data_processing, and the results are stored in the folder Processed_data.
When constructing global means, we load as many datasets as possible from the cloud. This step is done in the notebook Cloud_averaging, and functions used for averaging are stored in global_annual_means.py. Everything that cannot be loaded from the cloud is downloaded instead, either with wget scripts generated from https://esgf-node.llnl.gov/projects/cmip6/ or from wget scripts generated in the notebook wget_and_averaging. Global means of downloaded files are also constructed in this notebook.

In the next step we read and save all calendar and branch information from the metadata. This is done with the notebooks branch_years and model_calendars.

Finally, we use all this information to construct global annual anomalies in the notebook Global_anomalies, with help from the functions stored in processing_functions.py.

## Step 3: Estimate forcing from the global annual anomalies
First we estimate all parameters for the abrupt-4xCO2 experiments in the notebook abruptCO2exp_linresponses_allmembers, using functions defined in estimation.py. The parameters from this experiment are saved in Estimates, and loaded in the notebook EstimateTransientERF where we estimate the forcing for experiments with a time-varying forcing, such as the 1pctCO2, historical and SSP experiments.

Our forcing estimates are compared to fixed-SST forcing estimates, computed in the notebook fixedSSTforcing for abrupt-4xCO2 and in FixedSST_transientERF for the historical + ssp245 experiment. 

All results are saved in Estimates, and analysed further in other notebooks.  

Member mean transient forcing for each model (the data behind Figure 3) can be found in the following files, located in Estimates:  
[1pctCO2 ERF](Estimates/member_mean_ERF_1pctCO2.csv)  
[historical ERF](Estimates/member_mean_ERF_historical.csv)  
[SSP1-1.9 ERF](Estimates/member_mean_ERF_ssp119.csv)  
[SSP1-2.6 ERF](Estimates/member_mean_ERF_ssp126.csv)  
[SSP2-4.5 ERF](Estimates/member_mean_ERF_ssp245.csv)  
[SSP3-7.0 ERF](Estimates/member_mean_ERF_ssp370.csv)  
[SSP5-8.5 ERF](Estimates/member_mean_ERF_ssp585.csv)  

These are averages over the forcing estiamtes for individual members found in the folder [Estimates/Transient_forcing_estimates](Estimates/Transient_forcing_estimates)