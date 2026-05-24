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
from reward.process_reward_model import ProcessRewardModel


class PRMTrainer(Trainer):
  def __init__(self, model: ProcessRewardModel, config: STRIDEConfig, max_steps: int = 100_000) -> None:
    super().__init__(model, config, max_steps)

  def compute_loss(self, batch: Dict[str, Any]) -> Tensor:
    step_hidden = batch["step_hidden"].to(self.device)
    graph_nodes = batch["graph_nodes"].to(self.device)
    step_labels = batch["step_labels"].to(self.device)

    mask = batch.get("mask", None)
    if mask is not None:
      mask = mask.to(self.device)

    out = self.model(step_hidden, graph_nodes, mask)

    valid = step_labels >= 0
    loss = F.binary_cross_entropy_with_logits(
      out.step_scores[valid].float(),
      step_labels[valid].float(),
    )

    return loss


if __name__ == "__main__":
  config = STRIDEConfig.load()

  print("── Building PRMTrainer ─────────────────")
  model = ProcessRewardModel(config)
  trainer = PRMTrainer(model, config)
  print(f"  device:     {trainer.device}")
  print(f"  parameters: {trainer.count_parameters():,}")

  cfg = config.prm
  B = config.training.batch_size
  n_steps = cfg.max_step_len
  max_nodes = cfg.max_graph_nodes

  def make_batch():
    step_hidden = torch.randn(B, n_steps, cfg.d_model)
    graph_nodes = torch.randn(B, max_nodes, cfg.d_model)
    step_labels = torch.randint(0, 2, (B, n_steps))
    step_labels[:, -8:] = -1
    return {
      "step_hidden": step_hidden,
      "graph_nodes": graph_nodes,
      "step_labels": step_labels,
    }

  print("\n── Training steps ──────────────────────")
  grad_accum = config.training.gradient_accumulation_steps
  losses = []

  for i in range(grad_accum * 2):
    result = trainer.train_step(make_batch())
    if result is not None:
      losses.append(result)
      print(f"  step {trainer.step}: loss {result:.4f}")

  assert trainer.step == 2, f"Expected 2 steps, got {trainer.step}"
  assert all(0.0 < l < 10.0 for l in losses), "Losses out of expected range"
  print(f"  ✓ {trainer.step} optimizer steps, losses look finite")

  print("\n✓ PRMTrainer test complete")