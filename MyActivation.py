import torch
import torch.nn as nn
from torch.autograd import Variable
class MyActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v):
        sig = nn.Sigmoid()
        act = sig(v)*100
        return act

