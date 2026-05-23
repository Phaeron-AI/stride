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


class ComplexityReasonerOutput(NamedTuple):
  loop_state: LoopState
  complexity_logits: Tensor
  trace_repr: Tensor


class ComplexityReasoner(STRIDETransformer):
  def __init__(self, config: STRIDEConfig) -> None:
    cfg = config.complexity_reasoner
    vocab = config.vocab

    super().__init__(
      vocab_size = vocab.vocab_size,
      d_model = cfg.d_model,
      n_heads = cfg.n_heads,
      n_kv_heads = cfg.n_heads // 2,
      d_head = cfg.d_head,
      n_layers = cfg.n_layers,
      d_ff = cfg.d_ff,
      n_blocks = cfg.n_layers // 4,
      max_loops = 2,
      max_seq_len = cfg.max_seq_len,
      rope_base = 10000,
      dropout = cfg.dropout,
      entropy_beta = 0.1,
    )

    self.ast_proj = nn.Linear(64, cfg.d_model, bias=False)

    self.complexity_head = nn.Sequential(
      nn.Linear(cfg.d_model, cfg.d_model // 2, bias=True),
      nn.SiLU(),
      nn.Linear(cfg.d_model // 2, cfg.complexity_classes, bias=True),
    )

  def forward(
    self,
    token_ids: Tensor,
    ast_features: Tensor,
    mask: Optional[Tensor] = None,
  ) -> ComplexityReasonerOutput:
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

    trace_repr = self.pool(loop_state)
    complexity_logits = self.complexity_head(trace_repr)

    return ComplexityReasonerOutput(
      loop_state = loop_state,
      complexity_logits = complexity_logits,
      trace_repr = trace_repr,
    )

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building ComplexityReasoner ─────────")
  model = ComplexityReasoner(config)
  print(f"  Parameters: {model.count_parameters():,}")
  print(f"  block_size: {model.block_size}")
  print(f"  max_loops:  {model.max_loops}")

  B, S = 2, 512
  token_ids = torch.randint(0, config.vocab.vocab_size, (B, S))
  ast_features = torch.randn(B, S, 64)

  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(token_ids, ast_features)

  C = config.complexity_reasoner.complexity_classes

  assert out.complexity_logits.shape == (B, C), \
    f"complexity_logits: expected ({B}, {C}), got {tuple(out.complexity_logits.shape)}"
  assert out.trace_repr.shape == (B, config.complexity_reasoner.d_model), \
    f"trace_repr: expected ({B}, {config.complexity_reasoner.d_model}), got {tuple(out.trace_repr.shape)}"

  print(f"  complexity_logits : {tuple(out.complexity_logits.shape)}  ✓")
  print(f"  trace_repr        : {tuple(out.trace_repr.shape)}  ✓")

  ce_losses = [torch.rand(B) for _ in range(model.max_loops)]
  loss = model.compute_loop_loss(out.loop_state, ce_losses)
  assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf"
  print(f"  loop loss         : {loss.item():.4f}  ✓ finite")

  if torch.cuda.is_available():
    model = model.cuda()
    token_ids = token_ids.cuda()
    ast_features = ast_features.cuda()
    with torch.no_grad():
      out = model(token_ids, ast_features)
    print(f"\n  GPU: {out.trace_repr.device}  ✓")
    print("  ✓ GPU forward pass clean")

  print("\n✓ ComplexityReasoner test complete")