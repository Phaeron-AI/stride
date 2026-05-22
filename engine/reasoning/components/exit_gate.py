import torch
import torch.nn as nn
from torch import Tensor

class ExitGate(nn.Module):
  def __init__(self, d_model: int)-> None:
    super().__init__()

    self.proj = nn.Linear(d_model, 1, bias=False)
  
  def forward(self, hidden: Tensor)-> Tensor:
    pooled = hidden.mean(dim=1)
    logit = self.proj(pooled).squeeze(-1)
    return torch.sigmoid(logit)