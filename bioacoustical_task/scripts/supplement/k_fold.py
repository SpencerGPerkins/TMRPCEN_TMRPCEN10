# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:59:33 2023

@author: Spencer perkins

Script: 10-Fold Cross validation set-up, BirdVox20k-> BirdVox2k
"""

import pandas as pd
from get_data import getDATA

audio_path, meta_path = getDATA.get_data(prob='detect', ds_size='BVfull')

# Create dataframe
df = pd.read_csv(meta_path)

# Dictionary for storing k-fold files
fold_dict = {'1st':[],
            '2nd':[],
            '3rd':[],
            '4th':[],
            '5th':[],
            '6th':[],
            '7th':[],
            '8th':[],
            '9th':[],
            '10th':[]}

# Keys for k-fold dictionary
keys = ['1st', '2nd',
        '3rd', '4th',
        '5th', '6th',
        '7th', '8th',
        '9th', '10th']

# Index control variables
start = 0
end = 200

# Loop adding 200 items for each fold
for i in range(10):
    files = df.iloc[start:end]
    fold_dict[keys[i]].append(files)
    start +=201
    end += 201
    files.to_csv('../data/k_fold_meta/'+keys[i]+'.csv')
    bird = fold_dict[keys[i]][0]['hasbird'].sum()
    no_bird = len(fold_dict[keys[i]][0])-bird
    percentage = bird/len(fold_dict[keys[i]][0])
    print('Samples: %d, Positive Total: %d, Negative Total: %d, Percentage with bird: %.2f'
          % (len(fold_dict[keys[i]][0]), bird, no_bird, percentage))


    