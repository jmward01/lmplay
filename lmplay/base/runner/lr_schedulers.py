from .common import Component
from .common_runner import LMRunnerCommon
from lmplay.base.lrschedule import OptimizerWarmupLRScheduler, NoOpLRScheduler


class LRSchedulersComponent(Component):
  def __init__(self, mr: LMRunnerCommon):
    super().__init__()
    self.mr = mr

  def archive(self) -> dict:
    """Archive learning rate scheduler state.

    Saves the scheduler configuration (state_args) and state. Always returns
    scheduler states as a list for consistency with optimizer handling.

    Returns:
      dict with 'state_args' (scheduler config) and 'state' (list of scheduler states)
    """
    scheduler_states = [scheduler.state_dict() for scheduler in self.mr._lr_schedulers]

    return {
            'state_args': {},  # Schedulers don't have configurable state_args currently
            'state': scheduler_states
    }

  def advertise(self) -> dict:
    """Advertise learning rate scheduler construction parameters.

    Returns warmup scheduler parameters. Warmup is a one-time special case that occurs
    when optimizer state is missing/incompatible, allowing LR to ramp up gradually.

    Returns:
      dict with warmup parameters (warmup disabled by default)
    """
    return {
            'optimizer_warmup_fraction': None,  # Initial LR as fraction of target (e.g., 0.1)
            'optimizer_warmup_steps': None,  # Number of steps to warm up over
            'disable_optimizer_warmup': True,  # Warmup disabled by default
    }

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Construct learning rate scheduler(s) for optimizer(s).

    Creates one scheduler per optimizer in self.mr._optimizers. If warmup is configured and not disabled,
    creates OptimizerWarmupLRScheduler for ALL optimizers. Otherwise creates NoOpLRScheduler for all.

    Scheduler state is not loaded since schedulers are transient (full scheduler support
    will be implemented later with chaining, persistence, etc.).

    Args:
      construction_args: Scheduler construction arguments (optimizer_warmup_fraction,
                        optimizer_warmup_steps, disable_optimizer_warmup)
      state_args: Saved scheduler state arguments (unused for now)
      state: Saved scheduler states (unused for now)
    """
    # Must have optimizers constructed
    if not self.mr.for_train:
      #Nothing to construct if we aren't training
      return
    if not self.mr._optimizers:
      raise RuntimeError("Optimizers must be constructed before LR schedulers")

    # Extract warmup parameters from construction_args
    disable_optimizer_warmup = construction_args.get('disable_optimizer_warmup', True) if construction_args else True
    optimizer_warmup_fraction = construction_args.get('optimizer_warmup_fraction', None) if construction_args else None
    optimizer_warmup_steps = construction_args.get('optimizer_warmup_steps', None) if construction_args else None

    # Determine if we should create warmup scheduler
    # Create for all optimizers if: not disabled AND has warmup config
    create_warmup = (
            not disable_optimizer_warmup and
            optimizer_warmup_fraction is not None and
            optimizer_warmup_steps is not None
    )

    # Create schedulers for each optimizer
    lr_schedulers = []
    for optimizer in self.mr._optimizers:
      if create_warmup:
        # Warmup scheduler for this optimizer
        lr_schedulers.append(OptimizerWarmupLRScheduler(
                optimizer,
                steps=optimizer_warmup_steps,
                initial_fraction=optimizer_warmup_fraction))
      else:
        # No-op scheduler
        lr_schedulers.append(NoOpLRScheduler(optimizer))

    self.mr._lr_schedulers = lr_schedulers
