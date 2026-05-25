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


class PRMOutput(NamedTuple):
  step_scores: Tensor
  dim_scores: Tensor


class SelfAttentionLayer(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, max_seq_len: int, dropout: float) -> None:
    super().__init__()

    self.pre_attn_norm = RMSNorm(d_model)
    self.attn = GQA(
      d_model, n_heads, n_heads // 2,
      d_head, max_seq_len, 10000, dropout
    )
    self.pre_ffn_norm = RMSNorm(d_model)
    self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    x = x + self.dropout(self.attn(self.pre_attn_norm(x), mask))
    x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))
    return x


class CrossAttentionLayer(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()

    self.scale = d_model ** -0.5

    self.query_norm = RMSNorm(d_model)
    self.context_norm = RMSNorm(d_model)

    self.query_proj = nn.Linear(d_model, d_model, bias=False)
    self.key_proj = nn.Linear(d_model, d_model, bias=False)
    self.value_proj = nn.Linear(d_model, d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

    self.pre_ffn_norm = RMSNorm(d_model)
    self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor, nodes: Tensor) -> Tensor:
    Q = self.query_proj(self.query_norm(x))
    K = self.key_proj(self.context_norm(nodes))
    V = self.value_proj(self.context_norm(nodes))

    scores = Q @ K.transpose(-2, -1) * self.scale
    attn = F.softmax(scores, dim=-1)
    out = self.out_proj(attn @ V)

    x = x + self.dropout(out)
    x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))

    return x


class ProcessRewardModel(nn.Module):
  def __init__(self, config: STRIDEConfig) -> None:
    super().__init__()

    cfg = config.prm
    self.cross_attn_every = cfg.cross_attn_every

    self.input_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    self.node_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    self.layers = nn.ModuleList([
      CrossAttentionLayer(cfg.d_model, cfg.d_ff, cfg.dropout)
      if (i + 1) % self.cross_attn_every == 0
      else SelfAttentionLayer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_ff, cfg.max_step_len, cfg.dropout)
      for i in range(cfg.n_layers)
    ])

    self.final_norm = RMSNorm(cfg.d_model)

    self.score_head = nn.Sequential(
      nn.Linear(cfg.d_model, cfg.score_dimensions, bias=True)
    )

    self.register_buffer("score_weights", torch.tensor(cfg.score_weights, dtype=torch.float32))

  def forward(self, step_hidden: Tensor, graph_nodes: Tensor, mask: Optional[Tensor] = None) -> PRMOutput:
    x = self.input_proj(step_hidden)
    nodes = self.node_proj(graph_nodes)

    for layer in self.layers:
      if isinstance(layer, CrossAttentionLayer):
        x = layer(x, nodes)
      else:
        x = layer(x, mask)

    x = self.final_norm(x)
    dim_scores = self.score_head(x)
    step_scores = (dim_scores * self.score_weights).sum(dim=-1)

    return PRMOutput(step_scores=step_scores, dim_scores=dim_scores)

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building ProcessRewardModel ─────────")
  model = ProcessRewardModel(config)
  print(f"  Parameters: {model.count_parameters():,}")

  cfg = config.prm
  B = 2
  n_steps = cfg.max_step_len
  max_nodes = cfg.max_graph_nodes

  step_hidden = torch.randn(B, n_steps, cfg.d_model)
  graph_nodes = torch.randn(B, max_nodes, cfg.d_model)

  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(step_hidden, graph_nodes)

  SD = cfg.score_dimensions

  assert out.step_scores.shape == (B, n_steps), \
    f"step_scores: expected ({B}, {n_steps}), got {tuple(out.step_scores.shape)}"
  assert out.dim_scores.shape == (B, n_steps, SD), \
    f"dim_scores: expected ({B}, {n_steps}, {SD}), got {tuple(out.dim_scores.shape)}"
  assert out.step_scores.min() >= 0.0 and out.step_scores.max() <= 1.0, \
    "step_scores must be in [0, 1]"

  print(f"  step_scores : {tuple(out.step_scores.shape)}  ✓")
  print(f"  dim_scores  : {tuple(out.dim_scores.shape)}  ✓")
  print(f"  score range : [{out.step_scores.min():.3f}, {out.step_scores.max():.3f}]  ✓")

  if torch.cuda.is_available():
    model = model.cuda()
    step_hidden = step_hidden.cuda()
    graph_nodes = graph_nodes.cuda()
    with torch.no_grad():
      out = model(step_hidden, graph_nodes)
    print(f"\n  GPU: {out.step_scores.device}  ✓")
    print("  ✓ GPU forward pass clean")

  print("\n✓ ProcessRewardModel test complete")