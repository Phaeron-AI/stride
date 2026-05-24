import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
 
from pathlib import Path
from typing import Optional, NamedTuple
 
engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))
 
from config.global_config import STRIDEConfig
from reasoning.components import RMSNorm, GQA, SwiGLUFFN
from generation.syntax_verifier import SyntaxVerifier

class TestGeneratorOutput(NamedTuple):
  logits: Tensor
  error_logits: Tensor


class EncoderLayer(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    d_head: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float
  )-> None:
    
    self.pre_attn_norm = RMSNorm(d_model)
    self.attn = GQA(
      d_model, n_heads, n_heads // 2,
      d_head, max_seq_len, 10000, dropout
    )
    self.pre_ffn_norm = RMSNorm(d_model)
    self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor, mask: Optional[Tensor] = None)-> Tensor:
    x = x + self.dropout(self.attn(self.pre_attn_norm(x), mask))
    x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))
    return x
  
class DecoderLayer(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, max_seq_len: int, dropout: float)-> None:
    super().__init__()
    
