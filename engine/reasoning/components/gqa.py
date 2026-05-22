import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional

from reasoning.components.rope import RoPE

class GQA(nn.Module):
  def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: int, max_seq_len: int, rope_base :int = 10000, dropout: float = 0.1)-> None:
    super().__init__()

    assert n_heads % n_kv_heads == 0, \
      f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}"

    self.d_model = d_model
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.d_head = d_head
    self.max_seq_len = max_seq_len
    self.rope_base = rope_base

    self.n_groups = n_heads // n_kv_heads
    self.scale = d_head ** -0.5

    self.q_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
    self.k_proj = nn.Linear(d_model, n_kv_heads*d_head, bias=False)
    self.v_proj = nn.Linear(d_model, n_kv_heads*d_head, bias=False)
    self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    self.rope = RoPE(self.d_head, self.max_seq_len, self.rope_base)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: Tensor, mask: Optional[Tensor] = None)-> Tensor:
    B, seq_len, _ = x.shape

    q = self.q_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    k = self.k_proj(x).view(B, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
    v = self.v_proj(x).view(B, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

    q, k = self.rope(q, k, seq_len)

    k = k.repeat_interleave(self.n_groups, dim=1)
    v = v.repeat_interleave(self.n_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    if mask is not None:
      scores = scores + mask
    
    attention = torch.softmax(scores, dim=-1)
    attention = self.dropout(attention)
    out = torch.matmul(attention, v)
    out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)

    return self.o_proj(out)