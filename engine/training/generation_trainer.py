import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pathlib import Path
from typing import Dict, Any

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from config.global_config import STRIDEConfig
from training.base_trainer import Trainer
from generation.code_generation_head import CodeGenerationHead
from generation.test_generator import TestGenerator


class GenerationTrainer(Trainer):
  model: CodeGenerationHead
  def __init__(self, model: nn.Module, config: STRIDEConfig, max_steps: int = 100_000) -> None:
    super().__init__(model, config, max_steps)

  def compute_loss(self, batch: Dict[str, Any]) -> Tensor:
    if isinstance(self.model, CodeGenerationHead):
      return self._generation_loss(batch)
    elif isinstance(self.model, TestGenerator):
      return self._test_gen_loss(batch)
    else:
      raise ValueError(f"Unknown model type: {type(self.model)}")

  def _generation_loss(self, batch: Dict[str, Any]) -> Tensor:
    token_ids = batch["token_ids"].to(self.device)
    graph_nodes = batch["graph_nodes"].to(self.device)
    labels = batch["labels"].to(self.device)
    mask = batch.get("mask", None)
    if mask is not None:
      mask = mask.to(self.device)

    B = token_ids.shape[0]
    out = self.model(token_ids, graph_nodes, mask)

    shifted_logits = out.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    V = shifted_logits.shape[-1]
    valid = shifted_labels.reshape(-1) >= 0
    ce = F.cross_entropy(
      shifted_logits.reshape(-1, V)[valid],
      shifted_labels.reshape(-1)[valid],
    )

    model: CodeGenerationHead = self.model
    ce_losses = [torch.zeros(B, device=self.device) for _ in range(model.max_loops)]
    loop = model.compute_loop_loss(out.loop_state, ce_losses)

    return ce + 0.1 * loop

  def _test_gen_loss(self, batch: Dict[str, Any]) -> Tensor:
    src_ids = batch["src_ids"].to(self.device)
    tgt_ids = batch["tgt_ids"].to(self.device)
    labels = batch["labels"].to(self.device)
    error_labels = batch["error_labels"].to(self.device)

    src_mask = batch.get("src_mask", None)
    tgt_mask = batch.get("tgt_mask", None)
    if src_mask is not None:
      src_mask = src_mask.to(self.device)
    if tgt_mask is not None:
      tgt_mask = tgt_mask.to(self.device)

    out = self.model(src_ids, tgt_ids, src_mask, tgt_mask)

    shifted_logits = out.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    V = shifted_logits.shape[-1]
    valid_lm = shifted_labels.reshape(-1) >= 0
    lm_ce = F.cross_entropy(
      shifted_logits.reshape(-1, V)[valid_lm],
      shifted_labels.reshape(-1)[valid_lm],
    )

    EC = out.error_logits.shape[-1]
    valid_syn = error_labels.reshape(-1) >= 0
    syntax_ce = F.cross_entropy(
      out.error_logits.reshape(-1, EC)[valid_syn],
      error_labels.reshape(-1)[valid_syn],
    )

    return lm_ce + 0.1 * syntax_ce


if __name__ == "__main__":
  config = STRIDEConfig.load()

  B = 2
  grad_accum = config.training.gradient_accumulation_steps

  print("── CodeGenerationHead ──────────────────")
  gen_model = CodeGenerationHead(config)
  gen_trainer = GenerationTrainer(gen_model, config)
  print(f"  parameters: {gen_trainer.count_parameters():,}")

  cfg = config.generation
  S = 64
  V = config.vocab.vocab_size
  N = config.graph_builder.max_nodes

  for i in range(grad_accum):
    batch = {
      "token_ids":   torch.randint(0, V, (B, S)),
      "graph_nodes": torch.randn(B, N, cfg.graph_node_dim),
      "labels":      torch.randint(0, V, (B, S)),
    }
    batch["labels"][:, -4:] = -1
    result = gen_trainer.train_step(batch)
  print(f"  step {gen_trainer.step}: loss {result:.4f}  ✓")

  print("── TestGenerator ───────────────────────")
  tg_model = TestGenerator(config)
  tg_trainer = GenerationTrainer(tg_model, config)
  print(f"  parameters: {tg_trainer.count_parameters():,}")

  src_len = 64
  tgt_len = 32
  EC = config.syntax_verifier.syntax_error_classes

  for i in range(grad_accum):
    batch = {
      "src_ids":      torch.randint(0, V, (B, src_len)),
      "tgt_ids":      torch.randint(0, V, (B, tgt_len)),
      "labels":       torch.randint(0, V, (B, tgt_len)),
      "error_labels": torch.randint(0, EC, (B, tgt_len)),
    }
    batch["labels"][:, -4:] = -1
    batch["error_labels"][:, -4:] = -1
    result = tg_trainer.train_step(batch)
  print(f"  step {tg_trainer.step}: loss {result:.4f}  ✓")

  print("\n✓ GenerationTrainer test complete")