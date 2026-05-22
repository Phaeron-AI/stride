import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, List

from reasoning.components.rms_norm import RMSNorm

class BlockAttnRes(nn.Module):
  def __init__(self, d_model: int, n_blocks: int):
    super().__init__()

    self.d_model = d_model
    self.n_blocks = n_blocks

    self.pseudo_query = nn.Parameter(torch.zeros(d_model))
    self.key_norm = RMSNorm(d_model)

  def forward(self, hidden: Tensor, blocks: List[Tensor], partial: Tensor)-> Tensor:
    all_sources = blocks + [partial]

    if not all_sources:
      return hidden
    
    V = torch.stack(all_sources, dim=0)
    K = self.key_norm(V)

    logits = torch.einsum("d, nbd -> nb", self.pseudo_query, K)
    alpha = F.softmax(logits, dim=0)

    h_res = torch.einsum("nb, nbd -> bd", alpha, V)
    h_res = h_res.unsqueeze(1).expand_as(hidden)

    return hidden + h_res