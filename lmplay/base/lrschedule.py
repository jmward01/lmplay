from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class OptimizerWarmupLRScheduler(LRScheduler):
  """Learning rate scheduler that implements warmup for stable training.

  This scheduler gradually increases the learning rate from an initial fraction
  to the target learning rate over a specified number of steps. This helps
  prevent instability in the early stages of training.

  Args:
    optimizer: The optimizer to schedule
    steps: Number of warmup steps (default: 100)
    initial_fraction: Starting learning rate as fraction of target (default: 0.2)
  """

  def __init__(self, optimizer: Optimizer, steps: Optional[int] = 100, initial_fraction: Optional[float] = 0.2):
    steps = steps if steps else 40
    initial_fraction = initial_fraction if initial_fraction else 0.1

    self.increasing = initial_fraction < 1.0
    self.max_lrs = [group['lr'] for group in optimizer.param_groups]
    self.current_lrs = [lr * initial_fraction for lr in self.max_lrs]
    self.step_size = [(m - c) / steps for m, c in zip(self.max_lrs, self.current_lrs)]
    # for the initial call on the first batch.
    self.current_lrs = [c - s for c, s in zip(self.current_lrs, self.step_size)]
    super().__init__(optimizer)

  def get_lr(self):
    """Calculate and return the next learning rate values.

    Returns:
      List[float]: Learning rates for each parameter group
    """
    if self.increasing:
      next_lr = [min(m, c + s) for m, c, s in zip(self.max_lrs, self.current_lrs, self.step_size)]
    else:
      next_lr = [max(m, c + s) for m, c, s in zip(self.max_lrs, self.current_lrs, self.step_size)]
    self.current_lrs = next_lr
    return self.current_lrs


class NoOpLRScheduler(LRScheduler):
  """No-op learning rate scheduler that does nothing.

  Used as a placeholder when no learning rate scheduling is needed, but a scheduler
  object is required for consistency across all optimizers in multi-optimizer setups.
  Calling step() has no effect on learning rates.
  """

  def __init__(self, optimizer: Optimizer):
    super().__init__(optimizer)

  def get_lr(self):
    """Return unchanged learning rates.

    Returns:
      List[float]: Current learning rates for each parameter group (unchanged)
    """
    return [group['lr'] for group in self.optimizer.param_groups]