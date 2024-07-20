from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

# groud-truth Laplace distribution
def LaplaceDisp2Prob(Gt,maxdisp=192,m=1,n=9):
    N,H,W = Gt.shape
    b = 0.8
            
    Gt = torch.unsqueeze(Gt,1)
    disp = torch.arange(maxdisp,device=Gt.device)
    disp = disp.reshape(1,maxdisp,1,1).repeat(N,1,H,W)
    cost = -torch.abs(disp-Gt) / b

    return F.softmax(cost,dim=1)


def Adaptive_Multi_Modal_Cross_Entropy_Loss(x,disp,mask,maxdisp,m=1,n=9, top_k=9, epsilon=3, min_samples = 1):

    disp[disp>=192] = 0

    N,H,W = disp.shape
    patch_h, patch_w = m, n
    disp_unfold = F.unfold(F.pad(disp,(patch_w//2,patch_w//2,patch_h//2,patch_h//2), mode='reflect'),(patch_h,patch_w)).view(N, patch_h*patch_w,H, W)
    disp_unfold_clone = torch.clone(disp_unfold)
    
    mask_cluster = torch.zeros((N,patch_h*patch_w,patch_h*patch_w,H,W), device=disp.device).bool()
    for index in range(patch_h*patch_w):
        if index == 0:
            d_min = d_max = disp.unsqueeze(1)
        else:
            disp_unfold = disp_unfold * ~mask_cluster[:,index-1,...]
            d_min = d_max = torch.max(disp_unfold, dim=1, keepdim=True)[0]
        mask_cluster[:,index,...] = (disp_unfold>(d_min-epsilon).clamp(min=1e-6)) & (disp_unfold < (d_max+epsilon).clamp(max=maxdisp-1))
        while True:
            d_min = torch.min(disp_unfold*mask_cluster[:,index,...] + ~mask_cluster[:,index,...] * 192, dim=1, keepdim=True)[0]
            d_max = torch.max(disp_unfold*mask_cluster[:,index,...], dim=1, keepdim=True)[0]
            mask_new = (disp_unfold>(d_min-epsilon).clamp(min=0)) & (disp_unfold < (d_max+epsilon).clamp(max=maxdisp-1))
            if mask_new.sum() == mask_cluster[:,index,...].sum():
                break
            else:
                mask_cluster[:,index,...] = mask_new
    
    disp_cluster = torch.mean(disp_unfold_clone.unsqueeze(1).repeat(1,patch_h*patch_w,1,1,1)*mask_cluster, dim=2) * (patch_h*patch_w) / torch.sum(mask_cluster, dim=2).clamp(min=1)

    GT = torch.zeros((N, patch_h*patch_w, maxdisp, H, W), device=disp.device)
    for index in range(patch_h*patch_w):
        if index == 0:
            GT[:,index,...] = LaplaceDisp2Prob(disp,maxdisp)
        else:
            GT[:,index,...] = LaplaceDisp2Prob(disp_cluster[:,index,...],maxdisp)

    mask_cluster = torch.sum(mask_cluster, dim=2, keepdim=True)
    mask_cluster[mask_cluster < min_samples] = 0

    w_cluster = 0.2 / (mask_cluster.sum(dim=1, keepdim=True)-1).clamp(min=1) * mask_cluster
    w_cluster[:,0,...] += 0.8 - 0.2 / (mask_cluster.sum(dim=1 ,keepdim=False)-1).clamp(min=1)

    top_k_values, top_k_indices = torch.topk(w_cluster, k=top_k, dim=1)
    w_cluster.fill_(0)
    w_cluster.scatter_(dim=1, index=top_k_indices, src=top_k_values)
    w_cluster = w_cluster / w_cluster.sum(dim=1,keepdim=True).clamp(min=1)

    GT = (GT * w_cluster).sum(dim=1, keepdim=False)
    GT = GT.detach_()
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)

    loss =  - (GT[mask]*x[mask]).sum() / num

    return loss
