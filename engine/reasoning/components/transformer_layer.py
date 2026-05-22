import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, List, Tuple

from reasoning.components.rms_norm import RMSNorm
from reasoning.components.gqa import GQA
from reasoning.components.swiglu import SwiGLUFFN
from reasoning.components.block_attn_res import BlockAttnRes

class TransformerLayer(nn.Module):
  def __init__(
    self, 
    d_model: int, 
    n_heads: int, 
    n_kv_heads: int, 
    d_head: int, 
    d_ff: int, 
    n_blocks: int, 
    max_seq_len: int, 
    rope_base: int = 10000, 
    dropout: float = 0.1
  )-> None:
    super().__init__()

    self.attn_res = BlockAttnRes(d_model, n_blocks)
    self.ffn_res = BlockAttnRes(d_model, n_blocks)

    self.pre_attn_norm = RMSNorm(d_model)
    self.post_attn_norm = RMSNorm(d_model)

    self.pre_ffn_norm = RMSNorm(d_model)
    self.post_ffn_norm = RMSNorm(d_model)

    self.attn = GQA(
      d_model, n_heads, n_kv_heads, d_head, max_seq_len, rope_base, dropout
    )

    self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
  
  def forward(self, x: Tensor, blocks: List[Tensor], partial: Tensor, mask: Optional[Tensor] = None)-> Tuple[Tensor, Tensor]:
    x = self.attn_res(x, blocks, partial)

    residual = x
    x = self.pre_attn_norm(x)
    x = self.attn(x, mask)
    x = self.post_attn_norm(x)

    x = residual + x
    partial = partial + x.mean(dim=1)

    x = self.ffn_res(x, blocks, partial)

    residual = x
    x = self.pre_ffn_norm(x)
    x = self.ffn(x)
    x = self.post_ffn_norm(x)
    x = residual + x

    return x, partial