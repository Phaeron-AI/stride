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
from reasoning.base_transformer import STRIDETransformer
from reasoning.execution_reasoner import ExecutionReasoner
from reasoning.causal_reasoner import CausalReasoner
from reasoning.complexity_reasoner import ComplexityReasoner


class SpecialistTrainer(Trainer):
  model: STRIDETransformer

  def __init__(self, model: STRIDETransformer, config: STRIDEConfig, max_steps: int = 100_000) -> None:
    super().__init__(model, config, max_steps)

  def compute_loss(self, batch: Dict[str, Any]) -> Tensor:
    if isinstance(self.model, ExecutionReasoner):
      return self._execution_loss(batch)
    elif isinstance(self.model, CausalReasoner):
      return self._causal_loss(batch)
    elif isinstance(self.model, ComplexityReasoner):
      return self._complexity_loss(batch)
    else:
      raise ValueError(f"Unknown model type: {type(self.model)}")

  def _base_inputs(self, batch: Dict[str, Any]):
    token_ids = batch["token_ids"].to(self.device)
    ast_features = batch["ast_features"].to(self.device)
    mask = batch.get("mask", None)
    if mask is not None:
      mask = mask.to(self.device)
    return token_ids, ast_features, mask

  def _loop_loss(self, loop_state, B: int) -> Tensor:
    ce_losses = [torch.zeros(B, device=self.device) for _ in range(self.model.max_loops)]
    return self.model.compute_loop_loss(loop_state, ce_losses)

  def _execution_loss(self, batch: Dict[str, Any]) -> Tensor:
    token_ids, ast_features, mask = self._base_inputs(batch)
    step_labels = batch["step_labels"].to(self.device)

    out = self.model(token_ids, ast_features, mask)

    valid = step_labels >= 0
    bce = F.binary_cross_entropy_with_logits(out.step_scores[valid], step_labels[valid].float())
    loop = self._loop_loss(out.loop_state, token_ids.shape[0])

    return bce + 0.1 * loop

  def _causal_loss(self, batch: Dict[str, Any]) -> Tensor:
    token_ids, ast_features, mask = self._base_inputs(batch)
    edge_labels = batch["edge_labels"].to(self.device)

    out = self.model(token_ids, ast_features, mask)

    B, S, _, E = out.edge_logits.shape
    logits_flat = out.edge_logits.view(B * S * S, E)
    labels_flat = edge_labels.view(B * S * S)
    valid = labels_flat >= 0
    ce = F.cross_entropy(logits_flat[valid], labels_flat[valid])
    loop = self._loop_loss(out.loop_state, B)

    return ce + 0.1 * loop

  def _complexity_loss(self, batch: Dict[str, Any]) -> Tensor:
    token_ids, ast_features, mask = self._base_inputs(batch)
    complexity_label = batch["complexity_label"].to(self.device)

    out = self.model(token_ids, ast_features, mask)

    ce = F.cross_entropy(out.complexity_logits, complexity_label)
    loop = self._loop_loss(out.loop_state, token_ids.shape[0])

    return ce + 0.1 * loop


if __name__ == "__main__":
  config = STRIDEConfig.load()

  B = 2
  S = 32
  grad_accum = config.training.gradient_accumulation_steps

  def make_base_batch(seq_len):
    return {
      "token_ids":    torch.randint(0, config.vocab.vocab_size, (B, seq_len)),
      "ast_features": torch.randn(B, seq_len, 64),
    }

  print("── ExecutionReasoner ───────────────────")
  exec_model = ExecutionReasoner(config)
  exec_trainer = SpecialistTrainer(exec_model, config)
  print(f"  parameters: {exec_trainer.count_parameters():,}")

  for i in range(grad_accum):
    batch = make_base_batch(S)
    batch["step_labels"] = torch.randint(0, 2, (B, S))
    batch["step_labels"][:, -4:] = -1
    result = exec_trainer.train_step(batch)
  print(f"  step {exec_trainer.step}: loss {result:.4f}  ✓")

  print("── CausalReasoner ──────────────────────")
  causal_model = CausalReasoner(config)
  causal_trainer = SpecialistTrainer(causal_model, config)
  print(f"  parameters: {causal_trainer.count_parameters():,}")

  for i in range(grad_accum):
    batch = make_base_batch(S)
    batch["edge_labels"] = torch.randint(0, config.causal_reasoner.edge_types, (B, S, S))
    batch["edge_labels"][:, -4:, :] = -1
    result = causal_trainer.train_step(batch)
  print(f"  step {causal_trainer.step}: loss {result:.4f}  ✓")

  print("── ComplexityReasoner ──────────────────")
  complexity_model = ComplexityReasoner(config)
  complexity_trainer = SpecialistTrainer(complexity_model, config)
  print(f"  parameters: {complexity_trainer.count_parameters():,}")

  for i in range(grad_accum):
    batch = make_base_batch(S)
    batch["complexity_label"] = torch.randint(0, config.complexity_reasoner.complexity_classes, (B,))
    result = complexity_trainer.train_step(batch)
  print(f"  step {complexity_trainer.step}: loss {result:.4f}  ✓")

  print("\n✓ SpecialistTrainer test complete")