import torch
import torch.nn as nn
class Sigmoid100(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v):
        sig = nn.Sigmoid()
        act = sig(v)*100
        return act

