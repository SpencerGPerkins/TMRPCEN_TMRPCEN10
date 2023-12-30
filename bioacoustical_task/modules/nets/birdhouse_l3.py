# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:24:41 2023

@author: Spencer perkins

Module :  CNN model, a slightly modified version of L3 audio subnetwork
SUPPORTING 1 CHANNEL INPUT
"""

import torch
import torch.nn as nn
import time

class birdHouseL3S(nn.Module):
    def __init__(self):
        super(birdHouseL3S, self).__init__()

        """Four Convloutional block implementation of L3 with some
            modifications"""

        # Block one
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )
        self.pool_b1 = nn.MaxPool2d(2,2)

        # Block two
        self.conv3 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        self.pool_b2 = nn.MaxPool2d(2,2)
        self.drop_b2 = nn.Dropout(0.25)
        
        # Block three
        self.conv5 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                        )
        self.conv6 = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                        )
        self.pool_b3 = nn.MaxPool2d(2,2)
        self.drop_b3 = nn.Dropout(0.25)

        # Block 4
        self.conv7 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
        self.conv8 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )

        self.global_aver = nn.AvgPool2d((107,16))
        
        # Out layer
        self.fc = nn.Linear(256,1)                       


    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool_b1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool_b2(x)
        x = self.drop_b2(x)
        
        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool_b3(x)
        x = self.drop_b3(x)
        
        # Block 4
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = self.global_aver(x)

        x = self.fc(x.squeeze(-1).squeeze(-1))

        x= torch.sigmoid(x)
        x= torch.reshape(x, (-1,))

        return x