#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esra

The initial data download for labels (for each Lower Super Output Area - LSOA) need to be completed from the corresponding websites (as below). 
This script reads these downloaded files in, and computes the deciles from the input values that are used in training as labels. 

Census data: http://infusecp.mimas.ac.uk/ [corresponding variable descriptions are given in the paper, and in the script you can find the census variable codes used for each]
English indices of deprivation 2015: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015
Household income estimates (only for London): https://data.london.gov.uk/dataset/household-income-estimates-small-areas
ONS Postcode Directory August 2017 downloaded from https://ons.maps.arcgis.com/home/item.html?id=151e4a246b91c34178a55aab047413f29b [links may change]


"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


labels_census=pd.read_csv('../../data/raw-meta-data/wyorkshire-labels-census.csv')

### LSOA Level ### 
labels_lsoa_census=labels_census.loc[labels_census['GEO_TYP2']=='LSOADZ']

# derived values - general health
# compute perceentage of peoople with bad or very bad health
labels_lsoa_census.loc[:,'perc-lsoa-generalhealth-bad-or-verybad']=np.divide(np.add(labels_lsoa_census.loc[:,('F1777-p-generalhealth-verybad')],labels_lsoa_census.loc[:,('F1782-p-generalhealth-bad')]),labels_lsoa_census.loc[:,('F1778-p-generalhealth-total')])
#compute deciles - most deprived being 1
labels_lsoa_census.loc[:,'decile-lsoa-generalhealth-bad-or-verybad']=pd.qcut(labels_lsoa_census['perc-lsoa-generalhealth-bad-or-verybad'],10,labels=np.arange(10,0,-1))

# derived values - occupancy rating - compute percentages
labels_lsoa_census.loc[:,'perc-lsoa-occupancy-rating--1orless']=np.divide(labels_lsoa_census.loc[:,'F2375-hh-occup-rating-rooms--1orless'],labels_lsoa_census.loc[:,'F2376-hh-occup-rating-rooms-total'])
#compute deciles - most deprived being = 1
labels_lsoa_census.loc[:,'decile-lsoa-occupancy-rating--1orless']=pd.qcut(labels_lsoa_census['perc-lsoa-occupancy-rating--1orless'],10,labels=np.arange(10,0,-1))

# derived values - unemployment - compute percentages
labels_lsoa_census.loc[:,'perc-lsoa-hrp-unemployment-percent']=np.divide(labels_lsoa_census.loc[:,'F1399-p-econ-unemployed'],labels_lsoa_census.loc[:,'F1391-p-econ-total'])
#compute deciles - most deprived being = 1
labels_lsoa_census.loc[:,'decile-lsoa-hrp-unemployment']=pd.qcut(labels_lsoa_census['perc-lsoa-hrp-unemployment-percent'],10,labels=np.arange(10,0,-1))

# derived values - highest level of qualification - compute percentages
labels_lsoa_census.loc[:,'perc-lsoa-qual-below-level2']=np.divide(np.add(labels_lsoa_census.loc[:,('F187-p-highest-qual-no')],labels_lsoa_census.loc[:,('F188-p-highest-qual-level1')]),labels_lsoa_census.loc[:,('F186-p-highest-qual-total')])
#compute deciles - most deprived being = 1
labels_lsoa_census.loc[:,'decile-lsoa-below-level2']=pd.qcut(labels_lsoa_census['perc-lsoa-qual-below-level2'],10,labels=np.arange(10,0,-1))

# change label name for age
labels_lsoa_census=labels_lsoa_census.rename(columns={'F184-age-mean':'lsoa-age-mean', 'F185-age-median':'lsoa-age-median'})

exp_labels_lsoa_census=labels_lsoa_census[['GEO_CODE', 'GEO_TYP2','lsoa-age-mean', 'lsoa-age-median',\
       'perc-lsoa-generalhealth-bad-or-verybad',\
       'decile-lsoa-generalhealth-bad-or-verybad',\
       'perc-lsoa-occupancy-rating--1orless',\
       'decile-lsoa-occupancy-rating--1orless',\
       'perc-lsoa-hrp-unemployment-percent', 'decile-lsoa-hrp-unemployment',\
       'perc-lsoa-qual-below-level2',\
       'decile-lsoa-below-level2']]


exp_labels_lsoa_census.to_pickle('../../data/raw-meta-data/LONDON_CENSUS_LABELS_LSOA.p')


# merge deprivation indices at the LSOA level
lsoa_labels=pd.read_pickle('../../data/raw-meta-data/LONDON_CENSUS_LABELS_LSOA.p')

deprivation_lsoa=pd.read_csv('../../data/raw-meta-data/deprivation-domains-uk-2015.csv')

deprivation_lsoa=deprivation_lsoa.rename(columns={'Income Rank (where 1 is most deprived)': 'dep-income-rank-uk-lsoa', \
                                                  'Employment Rank (where 1 is most deprived)': 'dep-employment-rank-uk-lsoa', \
                                                  'Education, Skills and Training Rank (where 1 is most deprived)': 'dep-education-rank-uk-lsoa', \
                                                  'Health Deprivation and Disability Rank (where 1 is most deprived)': 'dep-health-rank-uk-lsoa', \
                                                  'Crime Rank (where 1 is most deprived)': 'dep-crime-rank-uk-lsoa', \
                                                  'Barriers to Housing and Services Rank (where 1 is most deprived)': 'dep-housing-rank-uk-lsoa', \
                                                  'Living Environment Rank (where 1 is most deprived)':'dep-liv-env-rank-uk-lsoa'})

deprivation_lsoa[['LSOA code (2011)','dep-income-rank-uk-lsoa','dep-employment-rank-uk-lsoa', 'dep-education-rank-uk-lsoa', \
                                                         'dep-health-rank-uk-lsoa', 'dep-crime-rank-uk-lsoa', 'dep-housing-rank-uk-lsoa', 'dep-liv-env-rank-uk-lsoa']]= \
                   deprivation_lsoa[['LSOA code (2011)','dep-income-rank-uk-lsoa','dep-employment-rank-uk-lsoa', 'dep-education-rank-uk-lsoa', \
                                                         'dep-health-rank-uk-lsoa', 'dep-crime-rank-uk-lsoa', 'dep-housing-rank-uk-lsoa', 'dep-liv-env-rank-uk-lsoa']]

lsoa_labels=pd.merge(left=lsoa_labels,right=deprivation_lsoa[['LSOA code (2011)','dep-income-rank-uk-lsoa','dep-employment-rank-uk-lsoa', 'dep-education-rank-uk-lsoa', \
                                                         'dep-health-rank-uk-lsoa', 'dep-crime-rank-uk-lsoa', 'dep-housing-rank-uk-lsoa', 'dep-liv-env-rank-uk-lsoa']], how='left',left_on='GEO_CODE',right_on='LSOA code (2011)')
lsoa_labels=lsoa_labels.drop('LSOA code (2011)',1)

lsoa_labels.loc[:,'dep-income-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-income-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-employment-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-employment-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-education-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-education-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-health-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-health-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-crime-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-crime-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-housing-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-housing-rank-uk-lsoa'],10,labels=np.arange(1,11,1))
lsoa_labels.loc[:,'dep-liv-env-decile-london-lsoa']=pd.qcut(lsoa_labels['dep-liv-env-rank-uk-lsoa'],10,labels=np.arange(1,11,1))


# not available for cities other than London
# merge modelled income and income deciles
mod_income_lsoa=pd.read_csv('modelled-household-income-estimates-lsoa2.csv')
mod_income_lsoa=mod_income_lsoa[['Code', 'Mean 2012/13', 'Median 2012/13']]
mod_income_lsoa=mod_income_lsoa.rename(columns={'Mean 2012/13':'mean-income-12-13-lsoa'})
mod_income_lsoa=mod_income_lsoa.rename(columns={'Median 2012/13':'median-income-12-13-lsoa'})

#compute deciles
mod_income_lsoa['mean-income-decile-london-lsoa']=pd.qcut(mod_income_lsoa['mean-income-12-13-lsoa'],10,labels=np.arange(1,11))
mod_income_lsoa['median-income-decile-london-lsoa']=pd.qcut(mod_income_lsoa['median-income-12-13-lsoa'],10,labels=np.arange(1,11))

lsoa_labels=pd.merge(left=lsoa_labels,right=mod_income_lsoa[['Code','mean-income-12-13-lsoa','mean-income-decile-london-lsoa','median-income-12-13-lsoa','median-income-decile-london-lsoa']],how='left',left_on='GEO_CODE',right_on='Code')

if np.array_equal(lsoa_labels['Code'],lsoa_labels['GEO_CODE']):
    lsoa_labels=lsoa_labels.drop('Code',1)
    

lsoa_labels.to_pickle('../../data/raw-meta-data/LONDON_ALL_LABELS_LSOA.p')

del  deprivation_lsoa, labels_census, labels_lsoa_census, lsoa_labels, mod_income_lsoa, deprivation_lsoa, exp_labels_lsoa_census

# merge all to postcode file for each of the cities (e.g. London in this case)
lsoa_labels =pd.read_pickle('../../data/raw-meta-data/LONDON_ALL_LABELS_LSOA.p')
# this is the postcode directory file [ONS Postcode Directory August 2017] merged with IMGID's we've used in our analysis
# each postcode location will have an img_id corresponding to its id in our analyses [to be used in make_hdf5 and training as unique ids]
city_pd=pd.read_pickle('../../data/raw-meta-data/ONSPD_AUG_2017_LONDON_W_METADATA_IMGID.p')

city_pd=pd.merge(left=city_pd,right=lsoa_labels,how='left',left_on='lsoa11',right_on='GEO_CODE')
city_pd=city_pd.drop(['GEO_CODE', 'GEO_TYP2'],1)

city_pd.to_pickle('../../data/raw-meta-data/ONSPD_AUG_2017_LONDON_W_METADATA_IMGID.p')


