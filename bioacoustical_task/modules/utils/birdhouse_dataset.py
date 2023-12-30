# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:22:28 2023

@author: E214_PC

Module : Custom dataset for BirdVox-DCASE-20K dataset
"""

import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

from audio_utils import AudioUtils
from features import featureExtractor

class birdhouseDataset(Dataset):

    """Bird Detection Dataset"""

    def __init__(self, label_file : str, audio_dir : str,
                 features : str,
                 transform=T.ToTensor()):

        self.birdhouse_meta = label_file
        self.audio_dir = audio_dir
        self.features = features
        self.transform = transform

    def __len__(self):

        return len(self.birdhouse_meta)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.to_list()
        audio_path = self.audio_dir+str(self.birdhouse_meta['itemid'][idx])+'.wav'
        audio= AudioUtils.open(audio_path)
        S = featureExtractor.get_features(audio, features=self.features)
        data_point = torch.Tensor(S)
        data_point = data_point.unsqueeze(0)
        label = self.birdhouse_meta['hasbird'][idx]

        return data_point, label
