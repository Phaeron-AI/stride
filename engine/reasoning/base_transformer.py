import sys
import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from typing import Optional, NamedTuple, List

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from reasoning.components import (
  RMSNorm,
  TransformerLayer,
  ExitGate
)

class LoopState(NamedTuple):
  hidden: Tensor
  loop_outputs: List[Tensor]
  exit_probs: List[Tensor]
  survival: Tensor

class STRIDETransformer(nn.Module):
  def __init__(
    self,
    vocab_size:   int,
    d_model:      int,
    n_heads:      int,
    n_kv_heads:   int,
    d_head:       int,
    n_layers:     int,
    d_ff:         int,
    n_blocks:     int,
    max_loops:    int,
    max_seq_len:  int,
    rope_base:    int   = 10000,
    dropout:      float = 0.1,
    entropy_beta: float = 0.1,
  )-> None:
    super().__init__()

    assert n_layers % n_blocks == 0, \
      f"n_layers {n_layers} must be divisible by n_blocks {n_blocks}"
    
    self.d_model      = d_model
    self.n_layers     = n_layers
    self.n_blocks     = n_blocks
    self.block_size   = n_layers // n_blocks
    self.max_loops    = max_loops
    self.entropy_beta = entropy_beta

    self.embed = nn.Embedding(vocab_size, d_model)

    self.layers: nn.ModuleList = nn.ModuleList([
      TransformerLayer(
        d_model, 
        n_heads, 
        n_kv_heads, 
        d_head,
        d_ff, 
        n_blocks, 
        max_seq_len, 
        rope_base, 
        dropout
      ) for _ in range(n_layers)
    ])

    self.final_norm = RMSNorm(d_model)
    self.exit_gate = ExitGate(d_model)

    self.embed_dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

    for layer in self.layers:
      attn_res = getattr(layer, "attn_res")
      ffn_res = getattr(layer, "ffn_res")
      nn.init.zeros_(attn_res.pseudo_query)
      nn.init.zeros_(ffn_res.pseudo_query)
  
  def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
  def _one_loop_pass(self, hidden: Tensor, mask: Optional[Tensor] = None)-> Tensor:
    B = hidden.shape[0]

    blocks = []
    partial = torch.zeros(B, self.d_model, device=hidden.device, dtype=hidden.dtype)

    for layer_index, layer in enumerate(self.layers):
      hidden, partial = layer(hidden, blocks, partial, mask)

      if (layer_index + 1) % self.block_size == 0:
        blocks.append(partial.clone())
        partial = torch.zeros_like(partial)
    
    return self.final_norm(hidden)
  
  def loop_forward(self, token_ids: Tensor, mask: Optional[Tensor] = None) -> LoopState:
    hidden = self.embed_dropout(self.embed(token_ids))

    loop_outputs: List[Tensor] = []
    exit_probs: List[Tensor] = [] 
    
    survival = torch.ones(token_ids.shape[0], device=hidden.device, dtype=hidden.dtype)

    for step in range(self.max_loops):
      hidden = self._one_loop_pass(hidden, mask)
      loop_outputs.append(hidden)

      lambda_r = self.exit_gate(hidden)
      
      if step < self.max_loops - 1:
        p_r = lambda_r * survival
        exit_probs.append(p_r.mean(dim=1) if p_r.ndim > 1 else p_r)

        survival = survival * (1 - lambda_r)
      else:
        p_r = survival
        exit_probs.append(p_r.mean(dim=1) if p_r.ndim > 1 else p_r)

    return LoopState(
      hidden       = hidden,
      loop_outputs = loop_outputs,
      exit_probs   = exit_probs,
      survival     = survival,
    )
  
  def pool(self, loop_state: LoopState)-> Tensor:
    return loop_state.hidden.mean(dim=1)
  
  def pool_all_loops(self, loop_state: LoopState)-> List[Tensor]:
    return [h.mean(dim=1) for h in loop_state.loop_outputs]
  

  def compute_loop_loss(self, loop_state: LoopState, ce_losses: List[Tensor])-> Tensor:
    exit_probs = loop_state.exit_probs

    expected = sum(p_r*ce_r for p_r, ce_r in zip(exit_probs, ce_losses)).mean() # type: ignore

    p_stack = torch.stack(exit_probs, dim=0)  # (R, B)
    entropy = -(p_stack * (p_stack + 1e-8).log()).sum(dim=0).mean()
    
    return expected - self.entropy_beta* entropy
  
  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
 
  print("── Building STRIDETransformer ──────────")
 
  # Small config for fast shape testing
  model = STRIDETransformer(
    vocab_size   = 32000,
    d_model      = 128,
    n_heads      = 4,
    n_kv_heads   = 2,
    d_head       = 32,
    n_layers     = 8,
    d_ff         = 256,
    n_blocks     = 4,
    max_loops    = 2,
    max_seq_len  = 64,
    dropout      = 0.0,
    entropy_beta = 0.1,
  )
 
  print(f"  Parameters: {model.count_parameters():,}")
  print(f"  block_size: {model.block_size}")
 
  # ── Verify pseudo-queries are zeros ─────
  
  for i, layer in enumerate(model.layers):
    attn_res = getattr(layer, "attn_res")
    ffn_res = getattr(layer, "ffn_res")
    assert attn_res.pseudo_query.abs().sum() == 0, \
      f"Layer {i} attn_res pseudo_query not zero"
    assert ffn_res.pseudo_query.abs().sum() == 0, \
      f"Layer {i} ffn_res pseudo_query not zero"
  print("  ✓ All pseudo-queries initialized to zero")
 
  # ── Forward pass ────────────────────────
 
  B, S = 2, 64
  token_ids = torch.randint(0, 32000, (B, S))
 
  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    loop_state = model.loop_forward(token_ids)
 
  for r, h in enumerate(loop_state.loop_outputs):
    print(f"  loop {r+1} hidden: {tuple(h.shape)}")
 
  p_sum = sum(p.mean().item() for p in loop_state.exit_probs)
  print(f"  exit_probs sum:  {p_sum:.4f}  (should be 1.0)")
  assert abs(p_sum - 1.0) < 0.01, f"exit_probs don't sum to 1: {p_sum}"
 
  # ── Pooling ──────────────────────────────
 
  pooled     = model.pool(loop_state)
  all_pooled = model.pool_all_loops(loop_state)
  assert pooled.shape     == (B, 128), f"pool wrong shape: {pooled.shape}"
  assert len(all_pooled)  == 2,        f"wrong loop count: {len(all_pooled)}"
  assert all_pooled[0].shape == (B, 128)
  print(f"\n  pool()           → {tuple(pooled.shape)}")
  print(f"  pool_all_loops() → {len(all_pooled)} × {tuple(all_pooled[0].shape)}")
 
  # ── Loss ─────────────────────────────────
 
  ce_losses = [torch.rand(B) for _ in range(2)]
  loss = model.compute_loop_loss(loop_state, ce_losses)
  assert not torch.isnan(loss) and not torch.isinf(loss), "Loss NaN/Inf"
  print(f"\n  loop loss: {loss.item():.4f}  ✓ finite")
 
  # ── GPU ──────────────────────────────────
 
  if torch.cuda.is_available():
    model     = model.cuda()
    token_ids = token_ids.cuda()
    with torch.no_grad():
      loop_state = model.loop_forward(token_ids)
    pooled = model.pool(loop_state)
    print(f"\n  GPU: {pooled.device}  shape {tuple(pooled.shape)}  ✓")
 
  print("\n✓ Base transformer test complete")
 