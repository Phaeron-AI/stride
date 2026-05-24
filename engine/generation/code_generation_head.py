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
from reasoning.base_transformer import STRIDETransformer, LoopState
from reasoning.components import RMSNorm, SwiGLUFFN

class CodeGenerationOutput(NamedTuple):
  logits: Tensor
  loop_state: LoopState


class GraphCrossAttentionLayer(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float)-> None:
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
  
  def forward(self, x: Tensor, graph_nodes: Tensor)-> Tensor:
    Q = self.query_proj(self.query_norm(x))
    K = self.key_proj(self.context_norm(graph_nodes))
    V = self.value_proj(self.context_norm(graph_nodes))

    scores = Q @ K.transpose(-2, -1) * self.scale
    attn = F.softmax(scores, dim=-1)
    out = self.out_proj(attn @ V)

    x = x + self.dropout(out)
    x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))

    return x

class CodeGenerationHead(STRIDETransformer):
  def __init__(self, config: STRIDEConfig)-> None:
    cfg = config.generation
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
      max_seq_len = cfg.max_prompt_len + cfg.max_gen_len,
      rope_base = 10000,
      dropout = cfg.dropout,
      entropy_beta = 0.1,
    )

    self.graph_proj = nn.Linear(cfg.graph_node_dim, cfg.d_model)

    self.cross_attention_layers = nn.ModuleList([
      GraphCrossAttentionLayer(cfg.d_model, cfg.d_ff, cfg.dropout)
      for _ in range(cfg.n_layers // cfg.graph_cross_attn_every)
    ])

    self.lm_head = nn.Linear(cfg.d_model, 32000)
    # self.lm_head = self.embed.weight
  
  def forward(self, token_ids: Tensor, graph_nodes: Tensor, mask: Optional[Tensor] = None)-> CodeGenerationOutput:
    projected_nodes = self.graph_proj(graph_nodes)
    loop_state = self.loop_forward(token_ids, mask)

    hidden = loop_state.hidden
    for layer in self.cross_attention_layers:
      hidden = layer(hidden, projected_nodes)

    loop_state = loop_state._replace(hidden=hidden)
    
    logits = self.lm_head(hidden)

    return CodeGenerationOutput(
      logits=logits,
      loop_state=loop_state
    )
  
  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
  config = STRIDEConfig.load()
 
  print("── Building CodeGenerationHead ─────────")
  model = CodeGenerationHead(config)
  print(f"  Parameters:   {model.count_parameters():,}")
  print(f"  block_size:   {model.block_size}")
  print(f"  max_loops:    {model.max_loops}")
  print(f"  cross_layers: {len(model.cross_attention_layers)}")
 
  cfg = config.generation
  B, S = 2, 64
  token_ids = torch.randint(0, config.vocab.vocab_size, (B, S))
  graph_nodes = torch.randn(B, config.graph_builder.max_nodes, cfg.graph_node_dim)
 
  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(token_ids, graph_nodes)
 
  assert out.logits.shape == (B, S, config.vocab.vocab_size), \
    f"logits: expected ({B}, {S}, {config.vocab.vocab_size}), got {tuple(out.logits.shape)}"
 
  print(f"  logits     : {tuple(out.logits.shape)}  ✓")
  print(f"  loop_state : hidden {tuple(out.loop_state.hidden.shape)}  ✓")
 
  ce_losses = [torch.rand(B) for _ in range(model.max_loops)]
  loss = model.compute_loop_loss(out.loop_state, ce_losses)
  assert not torch.isnan(loss) and not torch.isinf(loss), "Loss NaN/Inf"
  print(f"  loop loss  : {loss.item():.4f}  ✓ finite")
 
  if torch.cuda.is_available():
    model = model.cuda()
    token_ids = token_ids.cuda()
    graph_nodes = graph_nodes.cuda()
    with torch.no_grad():
      out = model(token_ids, graph_nodes)
    print(f"\n  GPU: {out.logits.device}  ✓")
    print("  ✓ GPU forward pass clean")
 
  print("\n✓ CodeGenerationHead test complete")