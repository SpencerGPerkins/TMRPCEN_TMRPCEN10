#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:48:35 2023

@author: Spencer Perkins

Per-channel Energy Normalization frontend : trainable alpha, delta, r 

OPTIMIZED IIR filter process from : 
https://discuss.pytorch.org/t/for-loop-slows-training-down/167667
KFrank implementation


"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PCEN(nn.Module):
    """Apply Trainable Per-channel energy Normalization (train Alpha, delta, r)
    Args:
        n_bands : int, number of input frequency bands
        t_val : float, value for t -> s (smoothing coefficient)
        alpha : float, gain control
        delta : float, bias
        r : float,
        eps : float, small value to prevent division by 0
    """
    def __init__(self, n_bands: int=128, t_val : float= 2**8, alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6):
        super(PCEN, self).__init__()

        alpha = torch.log(torch.tensor(alpha))
        delta = torch.log(torch.tensor(delta))
        r = torch.log(torch.tensor(r))

        self.t_val = torch.as_tensor(t_val)
        self.alpha = nn.Parameter(alpha * torch.ones(n_bands))
        self.delta = nn.Parameter(delta * torch.ones(n_bands))
        self.r = nn.Parameter(r * torch.ones(n_bands))
        self.eps = torch.as_tensor(eps)

    def forward(self, x):

        # Calculate smoothing coefficient s
        s = (torch.sqrt(1 + 4* self.t_val**2) - 1) / (2 * self.t_val**2)
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()
        
        # Broadcast over channel dimension
        alpha = alpha.unsqueeze(1)
        delta = delta.unsqueeze(1)
        r = r.unsqueeze(1)
        
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
        pcen_out = pcen_.permute((0,1,3,2))

        return pcen_out
