import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SwiGLUFFN(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1)-> None:
    super().__init__()

    self.gate = nn.Linear(d_model, d_ff, bias=False)
    self.up = nn.Linear(d_model, d_ff, bias=False)
    self.down = nn.Linear(d_ff, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor)-> Tensor:
    return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))