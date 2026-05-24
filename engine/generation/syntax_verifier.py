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
from reasoning.graph_builder import EncoderLayer

class SyntaxVerifierOutput(NamedTuple):
  error_logits: Tensor
  has_error: Tensor

class SyntaxVerifier(nn.Module):
  def __init__(self, config: STRIDEConfig)-> None:
    super().__init__()
    cfg = config.syntax_verifier
    vocab = config.vocab

    self.embed = nn.Embedding(vocab.vocab_size, cfg.d_model)
    self.embed_dropout = nn.Dropout(cfg.dropout)
    
    self.encoder = nn.ModuleList([
      EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_head, 4 * cfg.d_model, cfg.max_seq_len, cfg.dropout)
      for _ in range(cfg.n_layers)
    ])

    self.final_norm = RMSNorm(d_model=cfg.d_model)
    self.error_head = nn.Linear(cfg.d_model, cfg.syntax_error_classes)
  
  def forward(self, token_ids: Tensor, mask: Optional[Tensor]= None)-> SyntaxVerifierOutput:
    x = self.embed_dropout(self.embed(token_ids))

    for layer in self.encoder:
      x = layer(x, mask)
    
    x = self.final_norm(x)
    error_logits = self.error_head(x)

    has_error = (error_logits.argmax(dim=-1) != 0).float()

    return SyntaxVerifierOutput(
      error_logits=error_logits, 
      has_error=has_error
    )
  
  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
  config = STRIDEConfig.load()
 
  print("── Building SyntaxVerifier ─────────────")
  model = SyntaxVerifier(config)
  print(f"  Parameters: {model.count_parameters():,}")
 
  cfg = config.syntax_verifier
  B, S = 2, cfg.max_seq_len
  token_ids = torch.randint(0, config.vocab.vocab_size, (B, S))
 
  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(token_ids)
 
  EC = cfg.syntax_error_classes
 
  assert out.error_logits.shape == (B, S, EC), \
    f"error_logits: expected ({B}, {S}, {EC}), got {tuple(out.error_logits.shape)}"
  assert out.has_error.shape == (B, S), \
    f"has_error: expected ({B}, {S}), got {tuple(out.has_error.shape)}"
  assert out.has_error.dtype == torch.float32, \
    f"has_error must be float32, got {out.has_error.dtype}"
 
  print(f"  error_logits : {tuple(out.error_logits.shape)}  ✓")
  print(f"  has_error    : {tuple(out.has_error.shape)}  ✓")
  print(f"  error rate   : {out.has_error.mean().item():.3f}")
 
  if torch.cuda.is_available():
    model = model.cuda()
    token_ids = token_ids.cuda()
    with torch.no_grad():
      out = model(token_ids)
    print(f"\n  GPU: {out.error_logits.device}  ✓")
    print("  ✓ GPU forward pass clean")
 
  print("\n✓ SyntaxVerifier test complete")