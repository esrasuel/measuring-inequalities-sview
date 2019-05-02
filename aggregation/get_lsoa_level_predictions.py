#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esrasuel

Aggregating image level predictions to LSOA level predictions for measurement of performance. 

"""


import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import sklearn.metrics

 
city_name='wyorkshire'

variables=['decile-lsoa-generalhealth-bad-or-verybad', \
       'decile-lsoa-occupancy-rating--1orless', \
       'decile-lsoa-hrp-unemployment',\
       'decile-lsoa-below-level2',\
       'dep-health-decile-london-lsoa', \
       'dep-housing-decile-london-lsoa', \
       'dep-employment-decile-london-lsoa',\
       'dep-education-decile-london-lsoa',\
       'dep-income-decile-london-lsoa',\
       'dep-crime-decile-london-lsoa',\
       'dep-liv-env-decile-london-lsoa'\
        ]     


titles=['Self-reported health', 'Occupancy rating', 'Unemployment', 'Education', 'Health deprivation and disability', \
'Barriers to housing and services', 'Employment deprivation', 'Education deprivation', \
'Income deprivation', 'Crime deprivation', 'Living environment deprivation']

exper = ['train']

x=list()
y=list()
size=list()

# For testing measures only
n=0
k=0
for n in range(len(titles)):
    h0 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_0_out_of_5_folds_h5_vals.csv'.format(city_name, exper[k], variables[n]), header=None, names=['h5_vals'])
    h1 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_1_out_of_5_folds_h5_vals.csv'.format(city_name, exper[k], variables[n]), header=None, names=['h5_vals'])
    h2 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_2_out_of_5_folds_h5_vals.csv'.format(city_name, exper[k], variables[n]), header=None, names=['h5_vals'])
    h3 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_3_out_of_5_folds_h5_vals.csv'.format(city_name, exper[k], variables[n]), header=None, names=['h5_vals'])        
    h4 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_4_out_of_5_folds_h5_vals.csv'.format(city_name, exper[k], variables[n]), header=None, names=['h5_vals'])                        

    pred0 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_0_out_of_5_folds_predictions.csv'.format(city_name, exper[k], variables[n]))
    pred1 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_1_out_of_5_folds_predictions.csv'.format(city_name, exper[k], variables[n]))
    pred2 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_2_out_of_5_folds_predictions.csv'.format(city_name, exper[k], variables[n]))
    pred3 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_3_out_of_5_folds_predictions.csv'.format(city_name, exper[k], variables[n]))        
    pred4 = pd.read_csv('../../analysis/vmap-binomial-predictions/{}_{}_{}_4_out_of_5_folds_predictions.csv'.format(city_name, exper[k], variables[n]))                        
    
    preds0 = pd.concat([pred0, h0], axis=1, join_axes=[pred0.index])
    preds1 = pd.concat([pred1, h1], axis=1, join_axes=[pred1.index])
    preds2 = pd.concat([pred2, h2], axis=1, join_axes=[pred2.index])
    preds3 = pd.concat([pred3, h3], axis=1, join_axes=[pred3.index])
    preds4 = pd.concat([pred4, h4], axis=1, join_axes=[pred4.index])
    
    preds_all = pd.concat([preds0, preds1, preds2, preds3, preds4])
    del preds0, preds1, preds2, preds3, preds4
    del h0,h1,h2,h3,h4
    del pred0, pred1, pred2, pred3, pred4


    preds_all.loc[:,'decile-h5-predicted']=pd.qcut(preds_all['h5_vals'],10,labels=np.arange(0,10,1))

    
    preds_all['predicted']=preds_all['predicted']+1
    preds_all[variables[n]]=preds_all[variables[n]]+1  
    preds_all['decile-h5-predicted']=preds_all['decile-h5-predicted']+1

    #print(variables[n])
    # a few different approaches for aggregating to LSOA level predictions, the Scientific Reports paper used the 'pred-lsoa-h5-mean' method.
    if n < 11:
        preds_dummy = preds_all.copy()
        votmaj = lambda x: np.mean(np.argwhere(np.bincount(x)==np.bincount(x).max()))
        preds_dummy['pred-lsoa-mean'] = np.rint(preds_dummy['predicted'].groupby(preds_dummy['lsoa11']).transform('mean'))
        preds_dummy['pred-lsoa-h5-mean'] = np.rint(preds_dummy['decile-h5-predicted'].groupby(preds_dummy['lsoa11']).transform('mean'))
        preds_dummy['pred-lsoa-median'] = preds_dummy['predicted'].groupby(preds_dummy['lsoa11']).transform('median')
        preds_dummy['pred-lsoa-majvote'] = preds_dummy['predicted'].groupby(preds_dummy['lsoa11']).transform(votmaj)
        preds_dummy=preds_dummy.drop(['img_id','pcd','oa11','predicted','h5_vals', 'decile-h5-predicted'],axis=1)
        preds_dummy=preds_dummy.drop_duplicates()
        
        preds_dummy.to_csv('../../analysis/aggregate_predictions/{}/{}-{}-lsoa-predictions-{}.csv'.format(city_name,city_name,exper[k],variables[n]),index=False)


