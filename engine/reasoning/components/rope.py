import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Tuple

class RoPE(nn.Module):
  def __init__(self, d_head: int, max_seq_len: int, base: int = 10000)-> None:
    super().__init__()

    self.d_head = d_head
    self.max_seq_len = max_seq_len
    self.base = base

    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    self.register_buffer("inv_freq", inv_freq)

    self._build_cache(max_seq_len)
  
  def _build_cache(self, seq_len: int):
    t = torch.arange(seq_len).to(self.inv_freq) # type: ignore
    freqs = torch.outer(t, self.inv_freq) # type: ignore
    emb = torch.cat([freqs, freqs], dim=-1)

    self.register_buffer("cos_cached", emb.cos())
    self.register_buffer("sin_cached", emb.sin())

  def _rotate_half(self, x: Tensor)-> Tensor:
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]

    return torch.cat([-x2, x1], dim=-1)
  
  def forward(self, q: Tensor, k: Tensor, seq_len: int)-> Tuple[Tensor, Tensor]:
    cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0) # type: ignore
    sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0) # type: ignore
    q_rot = q * cos + self._rotate_half(q) * sin
    k_rot = k * cos + self._rotate_half(k) * sin
    return q_rot, k_rot