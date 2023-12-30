# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:26:46 2023

@author: spencer perkins

Script : Averaging of 10-fold results
"""
import pandas as pd
from summary_stats import Averaging, STD

path = '../../results/exv8/'

# Output data
out_df = {'method':[],
          'K-fold Accuracy average':[],
          'K-fold Accuracy STD':[],
          'K-fold has_bird average':[],
          'K-fold no_bird average':[],
          'K-fold F1 Score average':[]}

# Methods
frontends = {'PCEN': ['pcen_all/', 'pcen_'],
              'MRPCEN': ['mrpcen_all/', 'mrpcen_'],
              'TMRPCEN_10': ['tmrpcen10_all/', 'tmrpcen10_gpu_'],
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
               '10'
               ]

# Compute summary statistics accross methods
for f in range(1):
    # Averaging data
    aver_df = {'averaged validation accuracy':[],
              'averaged has_bird accuracy': [],
              'averaged no_bird accuracy': [],
              'average f1 score':[]
              }

    out_df['method'].append(frontend_keys[f])

    for i in range(10):
        frontend_method = frontends[frontend_keys[f]]
        # Experimental results (training/validation) dataframe
        df = pd.read_csv(path+frontend_method[0]+frontend_method[1]+result_data[i]+'.csv')
        # Averaging
        overall_average = Averaging.overall(df['val_acc'])
        aver_df['averaged validation accuracy'].append(overall_average)
        has_bird, no_bird = Averaging.class_based(df['has_bird_acc'], df['no_bird_acc'])
        aver_df['averaged has_bird accuracy'].append(has_bird)
        aver_df['averaged no_bird accuracy'].append(no_bird)
        f1_average = Averaging.overall(df['F1'])
        aver_df['average f1 score'].append(f1_average)
    df = pd.DataFrame.from_dict(aver_df)
    df['fold'] = [(x+1) for x in range(10)]
    df.to_csv('../../results/stats/'+frontend_keys[f]+'.csv')
    # Overall summary statistics
    k_fold_aver = Averaging.overall(aver_df['averaged validation accuracy'])
    k_fold_std = STD.overall_std(aver_df['averaged validation accuracy'], k_fold_aver)
    k_fold_hasb_aver = Averaging.overall(aver_df['averaged has_bird accuracy'])
    k_fold_nb_aver = Averaging.overall(aver_df['averaged no_bird accuracy'])
    k_fold_f1_aver = Averaging.overall(aver_df['average f1 score'])

    # Update output dataframe
    out_df['K-fold Accuracy average'].append(k_fold_aver)
    out_df['K-fold Accuracy STD'].append(k_fold_std)
    out_df['K-fold has_bird average'].append(k_fold_hasb_aver)
    out_df['K-fold no_bird average'].append(k_fold_nb_aver)
    out_df['K-fold F1 Score average'].append(k_fold_f1_aver)

    print(f'Frontend method: {frontend_keys[f]}\n')
    print(f'Overall Average: {k_fold_aver:.4f}')
    print(f'Overall STD: {k_fold_std:.4f}')
    print(f'Overall has_bird Average: {k_fold_hasb_aver:.4f}')
    print(f'Overall no_bird Average: {k_fold_nb_aver:.4f}')
    print(f'Averaged F1 Score: {k_fold_f1_aver:.4f}')
    print('--------------------\n')

df = pd.DataFrame.from_dict(out_df)
df.to_csv(
    '../../results/stats/exv8/'+df['method'][0]+'_'+df['method'][1]+'_'+df['method'][2]+'_mini_comp.csv')
df.to_csv('../../results/stats/exv6/10folds_all.csv')
