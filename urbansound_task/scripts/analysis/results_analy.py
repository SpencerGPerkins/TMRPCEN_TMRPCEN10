# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:26:46 2023

@author: spencer perkins

Analysis Script : Analysis of experimental results
"""
import pandas as pd
from summary_stats import Averaging, STD

path = '../../results/exv2/'
# Output data
out_df = {'method':[],
          'Aver. Acc.':[],
          'STD':[],
          'F1 Macro':[]
          }

# Methods
frontends = {'Log-Mel': ['logmel_all/', 'logmel_'],
             'PCEN': ['pcen_all/', 'pcen_'], 
             'MRPCEN':['mrpcen_all/', 'mrpcen_'],
             'TMRPCEN10': ['tmrpcen10_all/', 'tmrpcen10_gpu_'],
             'TMRPCEN': ['tmrpcen_all/', 'tmrpcen_gpu_']
             }

# Frontend keys
frontend_keys = list(frontends.keys())

# File names (second part)
result_data = ['1',
               '2',
               '3',
               '4',
               '5',
               '6',
               '7',
               '8',
               '9',
               '10']

# Compute summary statistics accross methods
for f in range(5):
    # Averaging data
    aver_df = {'val acc':[],
              'F1 macro': []
              }

    out_df['method'].append(frontend_keys[f])

    for i in range(10):
        frontend_method = frontends[frontend_keys[f]]
        # Experimental results (training/validation) dataframe
        df = pd.read_csv(path+frontend_method[0]+frontend_method[1]+result_data[i]+'.csv')
        # Averaging
        overall_average = Averaging.overall(df['val_acc'])
        aver_df['val acc'].append(overall_average)
        f1mac = Averaging.overall(df['F1_macro'])
        aver_df['F1 macro'].append(f1mac)
    df = pd.DataFrame.from_dict(aver_df)
    df['fold'] = [(x+1) for x in range(10)]
    df.to_csv('../../results/stats/'+frontend_keys[f]+'.csv')
    
    # Overall summary statistics
    k_fold_aver = Averaging.overall(aver_df['val acc'])
    k_fold_std = STD.overall_std(aver_df['val acc'], k_fold_aver)
    k_fold_f1mac = Averaging.overall(aver_df['F1 macro'])

    # Update output dataframe
    out_df['Aver. Acc.'].append(k_fold_aver)
    out_df['STD'].append(k_fold_std)
    out_df['F1 Macro'].append(k_fold_f1mac)

    print(f'Frontend method: {frontend_keys[f]}\n')
    print(f'Overall Average: {k_fold_aver:.4f}')
    print(f'Overall STD: {k_fold_std:.4f}')
    print(f'Overall F1 Macro: {k_fold_f1mac:.4f}')
    print('--------------------\n')

df = pd.DataFrame.from_dict(out_df)
df.to_csv(
    '../../results/stats/'+df['method'][0]+'_'+df['method'][1]+'_mini_comp.csv')
