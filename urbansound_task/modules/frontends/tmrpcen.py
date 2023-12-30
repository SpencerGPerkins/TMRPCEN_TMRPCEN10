#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Spencer Perkins

TMRPCEN frontend : 128 s values per rate

"""

import torch
import torch.nn as nn

class TMRPCEN(nn.Module):
    
    """Trainable Multi-rate PCEN
    Args:
        n_bands : int, number of input frequency bands
        alpha : float, AGC exponent
        delta : float, DRC bias
        r : float, DRC exponent
        eps : float, small value to prevent division by 0
    """
    
    def __init__(self, n_bands: int=128, alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6):
        super(TMRPCEN, self).__init__()

        # Logs for trainable parameters
        s1 = torch.log(torch.abs(torch.normal(0.61803, 0.61803*0.01, size=(n_bands,))))
        s2 = torch.log(torch.abs(torch.normal(0.3904, 0.3904*0.01, size=(n_bands,))))
        s3 = torch.log(torch.abs(torch.normal(0.1174, 0.1174*0.01, size=(n_bands,))))
        s4 = torch.log(torch.abs(torch.normal(0.0606, 0.0606*0.01, size=(n_bands,))))
        s5 = torch.log(torch.abs(torch.normal(0.0308, 0.0308*0.01, size=(n_bands,))))
        s6 = torch.log(torch.abs(torch.normal(0.0155, 0.0155*0.01, size=(n_bands,))))
        s7 = torch.log(torch.abs(torch.normal(0.0078, 0.0078*0.01, size=(n_bands,))))
        s8 = torch.log(torch.abs(torch.normal(0.0039, 0.0039*0.01, size=(n_bands,))))
        s9 = torch.log(torch.abs(torch.normal(0.002, 0.002*0.01, size=(n_bands,))))
        s10 = torch.log(torch.abs(torch.normal(0.0001, 0.0001*0.01, size=(n_bands,))))
        
        
        alpha = torch.log(torch.tensor(alpha))
        delta = torch.log(torch.tensor(delta))
        r = torch.log(torch.tensor(r))

        self.s1 = nn.Parameter(s1)

        self.s2 = nn.Parameter(s2)
        self.s3 = nn.Parameter(s3)
        self.s4 = nn.Parameter(s4)
        self.s5 = nn.Parameter(s5)
        self.s6 = nn.Parameter(s6)
        self.s7 = nn.Parameter(s7)
        self.s8 = nn.Parameter(s8)
        self.s9 = nn.Parameter(s9)
        self.s10 = nn.Parameter(s10)
        
        self.alpha = nn.Parameter(alpha * torch.ones(n_bands))
        self.delta = nn.Parameter(delta * torch.ones(n_bands))
        self.r = nn.Parameter(r * torch.ones(n_bands))
        self.eps = torch.as_tensor(eps)

    def forward(self, x):

        # Exponentials of trainable parameters
        s_1 = self.s1.exp()
        s_2 = self.s2.exp()
        s_3 = self.s3.exp()
        s_4 = self.s4.exp()
        s_5 = self.s5.exp()
        s_6 = self.s6.exp()
        s_7 = self.s7.exp()
        s_8 = self.s8.exp()
        s_9 = self.s9.exp()
        s_10 = self.s10.exp()
        
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()

        # Broadcast over channel dimension
        alpha = alpha.unsqueeze(1)
        delta = delta.unsqueeze(1)
        r = r.unsqueeze(1)

        # Smoothing coefficient values
        s_vals = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10]

        # Storage for each PCEN computation
        layered_pcen = []
        
        # TMRPCEN
        # Compute the Smoothed filterbank
        for s in s_vals:
            # Smoother
            smoother = [x[..., 0]] # Initialize with first frame
            # print(f' smoother : {smoother[0].is_cuda}')
            for frame in range(1, x.shape[-1]):
                smoother.append((1-s)*smoother[-1]+s*x[..., frame])
            smoother = torch.stack(smoother, -1)

            # Reformulation for (E / (eps + smooth)**alpha +delta)**r - delta**r
            # Vincent Lostenlan

            smooth = torch.exp(-alpha*(torch.log(self.eps)+
                                     torch.log1p(smoother/self.eps)))          
            pcen_ = (x * smooth + delta)**r - delta**r

            # Store PCEN from current s value
            layered_pcen.append(pcen_)
        # Stack all computed PCEN 'layers'
        pcen_out = torch.stack(layered_pcen)
        pcen_out = torch.squeeze(pcen_out)
        
        # Explicitly delete computation variables
        del layered_pcen, s_vals, smoother, smooth, pcen_

        # Reshape [Channels, batch_size, frequency bands, time samples]
        pcen_out = torch.permute(pcen_out, (1,0,2,3))

        return pcen_out
