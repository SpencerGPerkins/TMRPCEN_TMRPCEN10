#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:48:35 2023

@author: Spencer Perkins

Multi-rate PCEN frontend : Trainable Alpha, delta, r 

OPTIMIZED IIR filter process from : 
https://discuss.pytorch.org/t/for-loop-slows-training-down/167667
KFrank implementation

"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MRPCEN(nn.Module):
    
    """Apply Multi-rate Per-channel energy Normalization (trainable alpha, delta, r)
    Args:
        n_bands : int, number of input frequency bands
        t_values : list, values for t -> s (smoothing coefficient)
        alpha : float, gain control
        delta : float, bias
        r : float,
        eps : float, small value to prevent division by 0
    """
    
    def __init__(self, n_bands: int=128, t_values : list=[], alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6):
        super(MRPCEN, self).__init__()

        alpha = torch.log(torch.tensor(alpha))
        delta = torch.log(torch.tensor(delta))
        r = torch.log(torch.tensor(r))

        self.t_values = t_values
        self.alpha = nn.Parameter(alpha * torch.ones(n_bands))
        self.delta = nn.Parameter(delta * torch.ones(n_bands))
        self.r = nn.Parameter(r * torch.ones(n_bands))
        self.eps = torch.as_tensor(eps)

    def forward(self, x):
        # x shape : [batch size, channels, frequency bands, time samples]
        t_tens_vals =[]
        for i in range(len(self.t_values)):
            t_tens_vals.append(torch.as_tensor(self.t_values[i]))
        s_values = []
        for t in t_tens_vals:
            s = (torch.sqrt(1 + 4* t**2) - 1) / (2 * t**2)
            s_values.append(torch.as_tensor(s))
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()
        
        # Broadcast over channel dimension
        alpha = alpha.unsqueeze(1)
        delta = delta.unsqueeze(1)
        r = r.unsqueeze(1)
        
        # Store pcen layers
        layered_pcen = []

        # Multi-rate PCEN
        for s in s_values:
            # Optimized IIR filter process, retrieved from Pytorch forum
            # https://discuss.pytorch.org/t/for-loop-slows-training-down/167667
            # KFrank implementation
            periodPrecompute = None
            nPrecompute = None
            vPrecompute = None
            period = 512
            n = x.shape[-1]
            s = s
            # print(s)
            if  vPrecompute is None  or  period != periodPrecompute  or  n != nPrecompute:
                p = (1 - s) ** torch.arange(n + 1, device = device)
            
                v = p.repeat(n).reshape(n + 1, n)[:-1].triu()
        
                vPrecompute = v
                nPrecompute = n
                periodPrecompute = period
            smoother = s * x @ vPrecompute + (1 - s) * x[..., 0, None] @ vPrecompute[None, 0]

            # Reformulation for (E / (eps + smooth)**alpha +delta)**r - delta**r
            # Vincent Lostenlan
            smooth = torch.exp(-alpha*(torch.log(self.eps)+
                                     torch.log1p(smoother/self.eps)))
            pcen_ = (x * smooth + delta)**r - delta**r
            layered_pcen.append(pcen_)
        pcen_out = torch.stack(layered_pcen)
        pcen_out = torch.squeeze(pcen_out)

        pcen_out = torch.permute(pcen_out, (1,0,2,3))

        return pcen_out
