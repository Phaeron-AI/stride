import sys
import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from typing import Optional, NamedTuple

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from config.global_config import STRIDEConfig
from reasoning.base_transformer import STRIDETransformer, LoopState


class CausalReasonerOutput(NamedTuple):
  loop_state: LoopState
  edge_logits: Tensor
  trace_repr: Tensor


class CausalReasoner(STRIDETransformer):
  def __init__(self, config: STRIDEConfig) -> None:
    cfg = config.causal_reasoner
    vocab = config.vocab

    super().__init__(
      vocab_size = vocab.vocab_size,
      d_model = cfg.d_model,
      n_heads = cfg.n_heads,
      n_kv_heads = cfg.n_heads // 2,
      d_head = cfg.d_head,
      n_layers = cfg.n_layers,
      d_ff = cfg.d_ff,
      n_blocks = cfg.n_layers // 6,
      max_loops = 3,
      max_seq_len = cfg.max_seq_len,
      rope_base = 10000,
      dropout = cfg.dropout,
      entropy_beta = 0.1,
    )

    self.ast_proj = nn.Linear(cfg.ast_feature_dim, cfg.d_model, bias=False)
    self.edge_head = nn.Linear(2 * cfg.d_model, cfg.edge_types)

  def forward(
    self,
    token_ids: Tensor,
    ast_features: Tensor,
    mask: Optional[Tensor] = None,
  ) -> CausalReasonerOutput:
    fused = self.embed_dropout(
      self.embed(token_ids) + self.ast_proj(ast_features)
    )

    loop_outputs = []
    exit_probs = []
    survival = torch.ones(token_ids.shape[0], device=fused.device, dtype=fused.dtype)

    hidden = fused
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

    loop_state = LoopState(
      hidden = hidden,
      loop_outputs = loop_outputs,
      exit_probs = exit_probs,
      survival = survival,
    )

    B, S, D = hidden.shape
    hi = hidden.unsqueeze(2).expand(B, S, S, D)
    hj = hidden.unsqueeze(1).expand(B, S, S, D)
    pairs = torch.cat([hi, hj], dim=-1)
    edge_logits = self.edge_head(pairs)

    trace_repr = self.pool(loop_state)

    return CausalReasonerOutput(
      loop_state = loop_state,
      edge_logits = edge_logits,
      trace_repr = trace_repr,
    )

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building CausalReasoner ─────────────")
  model = CausalReasoner(config)
  print(f"  Parameters: {model.count_parameters():,}")
  print(f"  block_size: {model.block_size}")
  print(f"  max_loops:  {model.max_loops}")

  B, S = 2, 512
  token_ids = torch.randint(0, config.vocab.vocab_size, (B, S))
  ast_features = torch.randn(B, S, config.causal_reasoner.ast_feature_dim)

  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(token_ids, ast_features)

  E = config.causal_reasoner.edge_types

  assert out.edge_logits.shape == (B, S, S, E), \
    f"edge_logits: expected ({B}, {S}, {S}, {E}), got {tuple(out.edge_logits.shape)}"
  assert out.trace_repr.shape == (B, config.causal_reasoner.d_model), \
    f"trace_repr: expected ({B}, {config.causal_reasoner.d_model}), got {tuple(out.trace_repr.shape)}"

  print(f"  edge_logits : {tuple(out.edge_logits.shape)}  ✓")
  print(f"  trace_repr  : {tuple(out.trace_repr.shape)}  ✓")

  ce_losses = [torch.rand(B) for _ in range(model.max_loops)]
  loss = model.compute_loop_loss(out.loop_state, ce_losses)
  assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf"
  print(f"  loop loss   : {loss.item():.4f}  ✓ finite")

  if torch.cuda.is_available():
    model = model.cuda()
    token_ids = token_ids.cuda()
    ast_features = ast_features.cuda()
    with torch.no_grad():
      out = model(token_ids, ast_features)
    print(f"\n  GPU: {out.trace_repr.device}  ✓")
    print("  ✓ GPU forward pass clean")

  print("\n✓ CausalReasoner test complete")