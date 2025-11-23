from .common import Component
from .common_runner import LMRunnerCommon
from lmplay.train.datasets.plan import get_step_names
from lmplay.stats import modelstats

class RunnerComponent(Component):
  def __init__(self, mr: LMRunnerCommon):
    super().__init__()
    self.mr = mr

  def archive(self) -> dict:
    """Archive runner-level hyperparameters and training state.

    Saves runner configuration (batch_size, validation_batch_size, grad_clip, include_prompts,
    validation_interval, save_interval) and training state (statistics).

    Returns:
      dict with 'state_args' (runner config) and 'state' (training state)
    """
    return {
            'state_args': {
                    'batch_size': self.mr.batch_size,
                    'validation_batch_size': self.mr.validation_batch_size,
                    'grad_clip': self.mr.grad_clip,
                    'include_prompts': self.mr.include_prompts,
                    'validation_interval': self.mr.validation_interval,
                    'save_interval': self.mr.save_interval,
            },
            'state': {
                    'stats': self.mr.model_stats.dump_dict(),
                    'step_stats': {name: stat.dump_dict() for name, stat in self.mr.step_stats.items()},
            }
    }

  def advertise(self) -> dict:
    """Advertise runner construction parameters.

    Returns the current runner configuration so it can be exposed in config files.

    Returns:
      dict of runner configuration parameters
    """
    return {
            'validation_batch_size': self.mr.validation_batch_size,
            'validation_interval': self.mr.validation_interval,
            'save_interval': self.mr.save_interval,
    }

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Construct runner-level configuration and training state.

    Sets runner-level configuration (batch_size, grad_clip, include_prompts) and initializes
    training state (statistics). Respects reset_history flag to drop stats.

    Args:
      construction_args: Runner construction arguments (batch_size, grad_clip, include_prompts,
                        reset_history, run_name, default_freeze, etc)
      state_args: Saved runner state arguments (can override construction_args)
      state: Runner state including stats and step_stats
    """
    # Set run name for statistics tracking
    # If no run_name provided, derive from training plan step names (curriculum constructed first)
    run_name = construction_args.get('run_name', '') if construction_args else ''

    if not run_name:
      run_name = '_'.join(get_step_names(self.mr.training_plan))

    self.mr.run_name = f"_{run_name}" if len(run_name) > 0 else ""

    # Initialize step tracking
    self.mr.step_stats = dict()

    # Get reset_history from construction_args
    reset_history = construction_args.get('reset_history', False) if construction_args else False

    # Initialize or restore model statistics
    if reset_history or not state or 'stats' not in state:
      # Fresh stats (no checkpoint or reset requested)
      self.mr.model_stats = modelstats.ModelStats(
              model_name=f"{self.mr._model.name}{self.mr.run_name}",
              basedir=self.mr.stats_dir)
    else:
      # Restore stats from checkpoint
      self.mr.model_stats = modelstats.ModelStats(model_name=f"{self.mr._model.name}{self.mr.run_name}",
                                                  basedir=self.mr.stats_dir, **state['stats'])

    # Restore step-level statistics from checkpoint (unless reset_history)
    if state and not reset_history:
      if 'step_stats' in state:
        for step_name, step_data in state['step_stats'].items():
          model_name = f"{self.mr._model.name}{self.mr.run_name}_step_{step_name}"
          self.mr.step_stats[step_name] = modelstats.ModelStats(model_name=model_name,
                                                                basedir=self.mr.stats_dir, **step_data)

    # Hyperparameters: state_args override construction_args if provided
    for param_name in ['batch_size', 'validation_batch_size', 'grad_clip', 'include_prompts', 'validation_interval',
                       'save_interval', 'default_freeze']:
      if state_args and param_name in state_args:
        setattr(self, param_name, state_args[param_name])
      elif construction_args and param_name in construction_args:
        setattr(self, param_name, construction_args[param_name])