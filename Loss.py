import torch
import torch.nn as nn
import torch.nn.functional as func


class CharbonnierLoss(nn.Module):

    def __init__(self):
        super(CharbonnierLoss,self).__init__()

    def forward(self,pre,gt):
        N = pre.shape[0]
        diff = torch.sum(torch.sqrt((pre - gt ).pow(2) + 0.001 **2)) / N
        return diff


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss,self).__init__()

    def forward(self,pre,gt):
        N = pre.shape[0]        
        diff = torch.sum((pre - gt ).pow(2)) / N
        return diff

class PsnrLoss(nn.Module):

    def __init__(self):
        super(PsnrLoss,self).__init__()

    def forward(self,pre,gt):
        N = pre.shape[0]        
        L2 = torch.mean((pre - gt).pow(2).view(N,-1), 1)
        psnr = torch.mean(10 * ((1.0 / L2).log10()))
        return psnr
