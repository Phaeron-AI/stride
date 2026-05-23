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
from reasoning.components import RMSNorm, GQA, SwiGLUFFN


class GraphBuilderOutput(NamedTuple):
  node_embeddings: Tensor
  node_logits: Tensor
  edge_logits: Tensor


class EncoderLayer(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    d_head: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float,
  ) -> None:
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


class ReasoningGraphBuilder(nn.Module):
  def __init__(self, config: STRIDEConfig) -> None:
    super().__init__()

    cfg = config.graph_builder
    self.max_nodes = cfg.max_nodes
    self.tokens_per_node = cfg.tokens_per_node

    self.input_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    self.encoder = nn.ModuleList([
      EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_ff, cfg.max_seq_len, cfg.dropout)
      for _ in range(cfg.n_layers)
    ])

    self.node_norm = RMSNorm(cfg.d_model)
    self.node_head = nn.Linear(cfg.d_model, cfg.node_types)
    self.edge_head = nn.Linear(2 * cfg.d_model, cfg.edge_types)

  def forward(self, hidden: Tensor, mask: Optional[Tensor] = None) -> GraphBuilderOutput:
    hidden = self.input_proj(hidden)

    for layer in self.encoder:
      hidden = layer(hidden, mask)

    B, _, d_model = hidden.shape
    nodes = hidden.view(B, self.max_nodes, self.tokens_per_node, d_model).mean(dim=2)
    nodes = self.node_norm(nodes)

    node_logits = self.node_head(nodes)

    B, N, D = nodes.shape
    ni = nodes.unsqueeze(2).expand(B, N, N, D)
    nj = nodes.unsqueeze(1).expand(B, N, N, D)
    edge_logits = self.edge_head(torch.cat([ni, nj], dim=-1))

    return GraphBuilderOutput(
      node_embeddings = nodes,
      node_logits = node_logits,
      edge_logits = edge_logits,
    )

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building ReasoningGraphBuilder ──────")
  model = ReasoningGraphBuilder(config)
  print(f"  Parameters:      {model.count_parameters():,}")
  print(f"  max_nodes:       {model.max_nodes}")
  print(f"  tokens_per_node: {model.tokens_per_node}")

  cfg = config.graph_builder
  B, S = 2, cfg.max_seq_len
  hidden = torch.randn(B, S, cfg.d_model)

  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(hidden)

  N = cfg.max_nodes
  NT = cfg.node_types
  ET = cfg.edge_types

  assert out.node_embeddings.shape == (B, N, cfg.d_model), \
    f"node_embeddings: expected ({B}, {N}, {cfg.d_model}), got {tuple(out.node_embeddings.shape)}"
  assert out.node_logits.shape == (B, N, NT), \
    f"node_logits: expected ({B}, {N}, {NT}), got {tuple(out.node_logits.shape)}"
  assert out.edge_logits.shape == (B, N, N, ET), \
    f"edge_logits: expected ({B}, {N}, {N}, {ET}), got {tuple(out.edge_logits.shape)}"

  print(f"  node_embeddings : {tuple(out.node_embeddings.shape)}  ✓")
  print(f"  node_logits     : {tuple(out.node_logits.shape)}  ✓")
  print(f"  edge_logits     : {tuple(out.edge_logits.shape)}  ✓")

  if torch.cuda.is_available():
    model = model.cuda()
    hidden = hidden.cuda()
    with torch.no_grad():
      out = model(hidden)
    print(f"\n  GPU: {out.node_embeddings.device}  ✓")
    print("  ✓ GPU forward pass clean")

  print("\n✓ GraphBuilder test complete")