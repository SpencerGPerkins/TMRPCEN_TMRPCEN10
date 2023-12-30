# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:57:20 2023

@author: Spencer perkins

Module : retrieve data for k-fold cross validation
"""
import pandas as pd


directories = {'audio_path':
               ['../../data/bird/wav/'],
               'meta_path':['../../data/bird/k_fold_meta_full/']
               }

class getKfoldDATA:

    def get_folded_data(val_fold, folds):
        """
        Parameters
        ----------
        val_fold : str
            Which fold to withold for validation.
        folds : str
            All folds

        Returns
        -------
        audio_dir : str
            Path to audio files.
        meta_trdata : pd.DataFrame
            Meta data for training data.
        meta_vldata : pd.dataframe
            Meta data for validation data.
        """

        audio_dir = directories['audio_path'][0]
        fold_path = directories['meta_path'][0]
        if val_fold in folds:
            tr_folds = folds
            tr_folds.remove(val_fold)
            print(f'\nTraining Folds: {tr_folds}')
            print(f'Validation Fold: {val_fold}\n-------------------------\n')
            training_lst = [pd.read_csv(fold_path+fold) for fold in tr_folds]
            meta_trdata = pd.concat(training_lst, ignore_index=True)
            meta_vldata = pd.read_csv(fold_path+val_fold)

            print('Training Samples: %d, Validation Samples: %d\n----------------------\n'% (len(meta_trdata), len(meta_vldata)))

            return (audio_dir, meta_trdata, meta_vldata)

        else:
            raise ValueError('Input shold be in number + suffix form + .csv -- e.g. 1st.csv, 2nd.csv')
