# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:03:42 2023

@author: Spencer perkins

Module : custom torch dataset for UrbanSound8K dataset
"""

import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

import librosa

from audio_utils import AudioUtils
from features import featureExtractor

class urbanhouseDataset(Dataset):
    
    """Urban Sound classification dataset"""
    
    def __init__(self, label_file : str, audio_dir : str,
                features : str, transform=T.ToTensor()):
        
        self.urban_meta = label_file
        self.urban_path = audio_dir
        self.features = features
        self.transform = transform
        
    def __len__(self):
        return len(self.urban_meta)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
            
        audio = AudioUtils.open(
            self.urban_path+str(self.urban_meta['itemid'][idx])
            )
        S = featureExtractor.get_features(audio, features=self.features)
        if len(S[1]) < 400: # Address inconsistencies in sample lens
            S = librosa.util.fix_length(S, size=400, axis=1, )
        data_point = torch.Tensor(S)
        data_point = data_point.unsqueeze(0)
        label = self.urban_meta['classID'][idx]
        
        return data_point, label
        
        