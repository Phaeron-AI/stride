import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-6):
    self.d_model = d_model
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d_model))
  
  def forward(self, x: Tensor)-> Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
    return self.weight * (x / rms)