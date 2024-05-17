from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np



# soft-argmax
class softargmax_estimator(nn.Module):
    def __init__(self, maxdisp):
        super(softargmax_estimator, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)
        out = torch.sum(x*disp,1, keepdim=True)
        out = torch.squeeze(out, 1)
        return out



# argmax
class argmax_estimator(nn.Module):
    def __init__(self, maxdisp):
        super(argmax_estimator, self).__init__()

    def forward(self, x):
        out = torch.argmax(x,1, keepdim=True)
        return out





# single-modal disparity estimator
class single_modal_estimator(nn.Module):
    def __init__(self, maxdisp):
        super(single_modal_estimator, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)
        index = torch.argmax(x,1,keepdim=True)
        mask = disp.repeat(x.size(0),1,x.size(2),x.size(3))
        mask2 = torch.arange(self.maxdisp+1,dtype=x.dtype,device=x.device).reshape([1,self.maxdisp+1,1,1]).repeat(x.size(0),1,x.size(2),x.size(3))

        x_diff_r = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device),\
                            append=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
        x_diff_l = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
        
        
        index_r = torch.gt(x_diff_r * torch.gt(mask2,index),0).int()
        index_r = torch.argmax(index_r,1,keepdim=True)-1
        
        index_l = torch.lt(x_diff_l * torch.le(mask,index),0).int()
        index_l = (self.maxdisp-1) - torch.argmax(torch.flip(index_l,[1]),1,keepdim=True)
        
        mask = torch.ge(mask,index_l) * torch.le(mask,index_r)
        x = x * mask.data
        x = x / torch.sum(x,1,keepdim=True)
        
        out = torch.sum(x*disp,1, keepdim=True)
        return out


# our dominant-modal disparity estimator
class dominant_modal_estimator(nn.Module):
    def __init__(self, maxdisp):
        super(dominant_modal_estimator, self).__init__()
        self.maxdisp = maxdisp


    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)

        x_blur = x.clone()
        x_blur = x_blur.permute(0,2,3,1).reshape(x.size(0),-1,x.size(1))
        k = 5
        kernel = torch.ones(x_blur.size(1),1,k,dtype=x_blur.dtype,device=x_blur.device) / k
        x_blur = F.conv1d(x_blur,kernel,padding='same',groups=x_blur.size(1))
        x_blur = x_blur.permute(0,2,1).reshape(x.shape)
        mask = modal_mask(x_blur)
        y = x * mask
        z = x - y
        x_blur = x_blur * (~mask)
        z = z * modal_mask(x_blur)

        valid = (torch.sum(y,1) >= torch.sum(z,1))
        valid = valid.to(torch.float32).unsqueeze(1)

        x = valid*y + (1-valid)*z
        x = x / torch.sum(x,1,keepdim=True)

        out = torch.sum(x*disp,1, keepdim=True)

        return out


def modal_mask(x):
    N, D, H, W = x.shape

    index = torch.argmax(x,1,keepdim=True)
    mask = torch.arange(D,dtype=x.dtype,device=x.device).reshape([1,D,1,1]).repeat(N,1,H,W)
    mask2 = torch.arange(D+1,dtype=x.dtype,device=x.device).reshape([1,D+1,1,1]).repeat(N,1,H,W)

    x_diff_r = torch.diff(x,dim=1,prepend=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device),\
                        append=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device))
    x_diff_l = torch.diff(x,dim=1,prepend=torch.ones(N,1,H,W,dtype=x.dtype,device=x.device))
    
    index_r = torch.gt(x_diff_r * torch.gt(mask2,index),0).int()
    index_r = torch.argmax(index_r,1,keepdim=True)-1
    
    index_l = torch.lt(x_diff_l * torch.le(mask,index),0).int()
    index_l = (D-1) - torch.argmax(torch.flip(index_l,[1]),1,keepdim=True)
    mask1 = torch.ge(mask,index_l) * torch.le(mask,index_r)

    r = torch.min(index_r-index,index-index_l)
    mask2 = torch.ge(mask,index-r) * torch.le(mask,index+r)

    valid = torch.abs(2*index-index_r-index_l)<3
    mask = valid * mask1 + ~valid * mask2
