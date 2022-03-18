# Code and data to be used for a paper about CMIP6 forcing

## Step 1: Find out which data are available for analysis
The work done in this step is stored in the folder Data_availability.
First we search through ESGF using pyclient to find all models with both piControl and abrupt-4xCO2 experiment.
For these models, we search through ESGF to find all available data for the experiments we are interested in.
Furthermore, we check which of these data are available through Google Cloud, and store this info for later use.

Later, we note that there are problems with some of the data:
It seems some data are removed from ESGF before downloading. Info about these data are stored in Data_availability/ESGF/esgf-missing_march3rd.json.
Some files are not used because they contain errors. Info about these are stored in Data_availability/ESGF/files_with_errors.json.

All data not available in GoogleCloud are downloaded. This includes also some files that we found to be in GoogleCloud, but which contained errors: cloud-buterrors.json.

## Step 2: Constructing global annual means and anomalies
The code for this work is stored in the folder Data_processing, and the results are stored in the folder Processed_data.
When constructing global means, we load as many datasets as possible from the cloud. This step is done in the notebook Cloud_averaging, and functions used for averaging are stored in global_annual_means.py. Everything that cannot be loaded from the cloud is downloaded instead, either with wget scripts generated from https://esgf-node.llnl.gov/projects/cmip6/ or from wget scripts generated in the notebook wget_and_averaging. Global means of downloaded files are also constructed in this notebook.

In the next step we read and save all calendar and branch information from the metadata. This is done with the notebooks branch_years and model_calendars.

Finally, we use all this information to construct global annual anomalies in the notebook Global_anomalies, with help from the functions stored in processing_functions.py.

## Step 3: Estimate forcing from the global annual anomalies

