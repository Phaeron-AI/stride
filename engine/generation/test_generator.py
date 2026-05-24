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


class DecoderLayer(nn.Module):
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

    self.scale = d_model ** -0.5

    self.pre_self_attn_norm = RMSNorm(d_model)
    self.self_attn = GQA(
      d_model, n_heads, n_heads // 2,
      d_head, max_seq_len, 10000, dropout
    )

    self.pre_cross_attn_norm = RMSNorm(d_model)
    self.encoder_norm = RMSNorm(d_model)

    self.query_proj = nn.Linear(d_model, d_model, bias=False)
    self.key_proj = nn.Linear(d_model, d_model, bias=False)
    self.value_proj = nn.Linear(d_model, d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

    self.pre_ffn_norm = RMSNorm(d_model)
    self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor, enc_out: Tensor, self_mask: Optional[Tensor] = None) -> Tensor:
    x = x + self.dropout(self.self_attn(self.pre_self_attn_norm(x), self_mask))

    Q = self.query_proj(self.pre_cross_attn_norm(x))
    K = self.key_proj(self.encoder_norm(enc_out))
    V = self.value_proj(self.encoder_norm(enc_out))

    scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
    attn_weights = F.softmax(scores, dim=-1)
    cross_out = self.out_proj(torch.matmul(attn_weights, V))

    x = x + self.dropout(cross_out)
    x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))

    return x


class TestGenerator(nn.Module):
  def __init__(self, config: STRIDEConfig) -> None:
    super().__init__()
    cfg = config.test_generator
    vcfg = config.syntax_verifier
    vocab = config.vocab

    self.src_embed = nn.Embedding(vocab.vocab_size, cfg.d_model)
    self.tgt_embed = nn.Embedding(vocab.vocab_size, cfg.d_model)
    self.embed_dropout = nn.Dropout(cfg.dropout)

    self.encoder = nn.ModuleList([
      EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_ff, cfg.max_combined_len, cfg.dropout)
      for _ in range(cfg.encoder_layers)
    ])
    self.enc_final_norm = RMSNorm(cfg.d_model)

    self.decoder = nn.ModuleList([
      DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_ff, cfg.max_test_len, cfg.dropout)
      for _ in range(cfg.decoder_layers)
    ])
    self.dec_final_norm = RMSNorm(cfg.d_model)

    self.lm_head = nn.Linear(cfg.d_model, vocab.vocab_size, bias=False)

    self.verifier_proj = nn.Linear(cfg.d_model, vcfg.d_model, bias=False)
    self.verifier = SyntaxVerifier(config)

  def forward(
    self,
    src_ids: Tensor,
    tgt_ids: Tensor,
    src_mask: Optional[Tensor] = None,
    tgt_mask: Optional[Tensor] = None,
  ) -> TestGeneratorOutput:
    src = self.embed_dropout(self.src_embed(src_ids))
    for layer in self.encoder:
      src = layer(src, src_mask)
    enc_out = self.enc_final_norm(src)

    tgt = self.embed_dropout(self.tgt_embed(tgt_ids))
    for layer in self.decoder:
      tgt = layer(tgt, enc_out, tgt_mask)
    dec_out = self.dec_final_norm(tgt)

    logits = self.lm_head(dec_out)

    projected = self.verifier_proj(dec_out)
    error_logits = self.verifier.error_head(projected)

    return TestGeneratorOutput(
      logits = logits,
      error_logits = error_logits,
    )

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building TestGenerator ──────────────")
  model = TestGenerator(config)
  print(f"  Parameters:      {model.count_parameters():,}")

  cfg = config.test_generator
  B = 2
  src_len = cfg.max_combined_len
  tgt_len = cfg.max_test_len

  src_ids = torch.randint(0, config.vocab.vocab_size, (B, src_len))
  tgt_ids = torch.randint(0, config.vocab.vocab_size, (B, tgt_len))

  print("\n── Forward pass ────────────────────────")
  with torch.no_grad():
    out = model(src_ids, tgt_ids)

  EC = config.syntax_verifier.syntax_error_classes

  assert out.logits.shape == (B, tgt_len, config.vocab.vocab_size), \
    f"logits: expected ({B}, {tgt_len}, {config.vocab.vocab_size}), got {tuple(out.logits.shape)}"
  assert out.error_logits.shape == (B, tgt_len, EC), \
    f"error_logits: expected ({B}, {tgt_len}, {EC}), got {tuple(out.error_logits.shape)}"

  print(f"  logits       : {tuple(out.logits.shape)}  ✓")
  print(f"  error_logits : {tuple(out.error_logits.shape)}  ✓")

  if torch.cuda.is_available():
    model = model.cuda()
    src_ids = src_ids.cuda()
    tgt_ids = tgt_ids.cuda()
    with torch.no_grad():
      out = model(src_ids, tgt_ids)
    print(f"\n  GPU: {out.logits.device}  ✓")
    print("  ✓ GPU forward pass clean")

  print("\n✓ TestGenerator test complete")