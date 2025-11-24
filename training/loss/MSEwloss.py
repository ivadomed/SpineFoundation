import torch.nn as nn
import torch.nn.functional as F

class MSEwloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target,weight):
        weight+=1
        return F.mse_loss(input, target, weight=weight)

