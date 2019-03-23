import torch
import torch.nn as nn
class Sigmoid(torch.nn.Module):
    def __init__(self, a=1, max = 10):
        super().__init__()
        self.a = a
        self.max = max

    def forward(self, v):
        sig = nn.Sigmoid()
        act = sig(self.a*v)*self.max
        return act

