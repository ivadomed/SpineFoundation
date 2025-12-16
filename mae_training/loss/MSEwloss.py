import torch.nn as nn
import torch.nn.functional as F
import torch 


class MSEwloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target,weight):
        if weight is None:
            weight = torch.ones_like(input)
        else:
            weight+=1
        return F.mse_loss(input, target, weight=weight)

