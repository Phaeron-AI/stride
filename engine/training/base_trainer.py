import sys
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from config.global_config import STRIDEConfig


class Trainer(ABC):
  def __init__(self, model: nn.Module, config: STRIDEConfig, max_steps: int = 100_000) -> None:
    self.model = model
    self.config = config
    self.step = 0
    self.accum_step = 0

    cuda_available = torch.cuda.is_available()
    self.device = torch.device(config.system.device if cuda_available else "cpu")
    self.model = self.model.to(self.device)

    self.optimizer = AdamW(
      self.model.parameters(),
      lr = config.training.learning_rate,
      betas = (0.9, 0.95),
      weight_decay = 0.1,
    )

    warmup = LinearLR(
      self.optimizer,
      start_factor = 1e-8,
      end_factor = 1.0,
      total_iters = config.training.warmup_steps,
    )
    decay = CosineAnnealingLR(
      self.optimizer,
      T_max = max(1, max_steps - config.training.warmup_steps),
    )
    self.scheduler = SequentialLR(
      self.optimizer,
      schedulers = [warmup, decay],
      milestones = [config.training.warmup_steps],
    )

  @abstractmethod
  def compute_loss(self, batch: Dict[str, Any]) -> Tensor:
    ...

  def train_step(self, batch: Dict[str, Any]) -> Optional[float]:
    self.model.train()
    cfg = self.config.training

    device_type = "cuda" if self.device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    with torch.autocast(device_type=device_type, dtype=dtype):
      loss = self.compute_loss(batch)

    scaled = loss / cfg.gradient_accumulation_steps
    scaled.backward()

    self.accum_step += 1

    if self.accum_step == cfg.gradient_accumulation_steps:
      return self.optimizer_step(loss.item())

    return None

  def optimizer_step(self, loss_val: float) -> float:
    nn.utils.clip_grad_norm_(
      self.model.parameters(),
      self.config.training.max_grad_norm,
    )
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
    self.accum_step = 0
    self.step += 1
    return loss_val

  def save_checkpoint(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
      "model":     self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "scheduler": self.scheduler.state_dict(),
      "step":      self.step,
    }, path)

  def load_checkpoint(self, path: str | Path) -> None:
    ckpt = torch.load(path, map_location=self.device)
    self.model.load_state_dict(ckpt["model"])
    self.optimizer.load_state_dict(ckpt["optimizer"])
    self.scheduler.load_state_dict(ckpt["scheduler"])
    self.step = ckpt["step"]

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


if __name__ == "__main__":
  import tempfile
  from config.global_config import STRIDEConfig

  config = STRIDEConfig.load()

  class DummyTrainer(Trainer):
    def compute_loss(self, batch):
      x = batch["x"].to(self.device)
      return (self.model(x) ** 2).mean()

  model = nn.Linear(8, 8)
  trainer = DummyTrainer(model, config)

  print("── Trainer smoke test ──────────────────")
  print(f"  device:     {trainer.device}")
  print(f"  parameters: {trainer.count_parameters():,}")
  print(f"  lr:         {config.training.learning_rate}")

  grad_accum = config.training.gradient_accumulation_steps
  losses = []

  for i in range(grad_accum * 3):
    batch = {"x": torch.randn(4, 8)}
    result = trainer.train_step(batch)
    if result is not None:
      losses.append(result)
      print(f"  step {trainer.step}: loss {result:.4f}")

  assert trainer.step == 3, f"Expected 3 optimizer steps, got {trainer.step}"
  assert len(losses) == 3, f"Expected 3 loss values, got {len(losses)}"
  print(f"  ✓ {trainer.step} optimizer steps completed")

  with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = Path(tmp) / "test.pt"
    trainer.save_checkpoint(ckpt_path)

    trainer2 = DummyTrainer(nn.Linear(8, 8), config)
    trainer2.load_checkpoint(ckpt_path)

    assert trainer2.step == trainer.step, "Step mismatch after load"
    print(f"  ✓ Checkpoint save/load verified (step={trainer2.step})")

  print("\n✓ Trainer test complete")