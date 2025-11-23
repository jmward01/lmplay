import os
import os.path
import json
import logging
from abc import ABC, abstractmethod
from shutil import copyfile
from typing import Optional, List, Dict, Any, Union, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lmplay.base.base_model import LMBase, logger
from lmplay.base.defaults import DEFAULT_LR
from lmplay.base.lrschedule import OptimizerWarmupLRScheduler, NoOpLRScheduler
from lmplay.base.optimizer import detect_freeze, get_default_weight_decay, categorize_parameters_by_weight_decay, \
  create_optimizer
from lmplay.base.utils import NopWith
from lmplay.stats import modelstats, utils
from lmplay.train.datasets.plan_configs import DEFAULT_PLANS
from lmplay.train.datasets.plan import get_step_names


class Section:
  """Wrapper for saveable/loadable sections of a runner.

  A Section encapsulates the operations for one saveable component:
  - archive(): Save current state
  - advertise(): Expose construction parameters
  - construct(): Build/rebuild the component

  Methods can be None, in which case nop versions are used.
  """

  def __init__(self, name: str, archive=None, advertise=None, construct=None):
    """Initialize a Section with optional method implementations.

    Args:
      name: Section name (e.g., 'model', 'optimizer')
      archive: Optional callable that returns {'state_args': {...}, 'state': {...}}
      advertise: Optional callable that returns construction args dict
      construct: Optional callable(construction_args, state_args, state) that returns object
    """
    self.name = name
    # Set up nop versions if not provided
    self._archive = archive if archive is not None else (lambda: {'state_args': {}, 'state': {}})
    self._advertise = advertise if advertise is not None else (lambda: {})
    self._construct = construct if construct is not None else (lambda ca, sa, s: None)

  def archive(self):
    """Call archive method, returns {'state_args': {...}, 'state': {...}}"""
    return self._archive()

  def advertise(self):
    """Call advertise method, returns construction args dict"""
    return self._advertise()

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Call construct method with all three parameters, returns constructed object"""
    return self._construct(construction_args, state_args, state)


class LMRunnerBase(ABC):
  """Abstract base class for model runners that handle training and inference.

  Runners wrap models and provide high-level functionality for:
  - Model initialization and device management
  - Training loops with gradient accumulation
  - Validation and generation
  - Checkpoint saving/loading
  - Statistics tracking
  - Optimizer and scheduler management

  Attributes:
    max_batch_size: Maximum batch size for gradient accumulation
    stats_dir: Directory for saving statistics files
  """

  def __init__(self, max_batch_size: int = 1, stats_dir="./out_gpt"):
    self.max_batch_size = max_batch_size
    # Runner-level training parameters with defaults (can be overridden during construct_runner)
    self.batch_size: int = 50  # Effective batch size for training
    self.validation_batch_size: int = 4  # Batch size for validation
    self.validation_interval: int = 100  # Run validation every N samples
    self.save_interval: int = 10000  # Save checkpoint every N samples
    self.grad_clip: Optional[float] = None  # Gradient clipping threshold
    self.include_prompts: bool = True  # Include prompts in loss calculation
    self.default_freeze: bool = False  # Whether to freeze model parameters by default
    # Model and optimization components
    self._model: Optional[LMBase] = None
    self._raw_model: Optional[LMBase] = None
    self._optimizers: Optional[List[Optimizer]] = None
    self.model_stats: Optional[modelstats.ModelStats] = None
    self.step_stats = dict()
    self.current_step = None
    self._model_args = None
    self._optimizer_args = None
    self._stats_dir = stats_dir
    self._lr_schedulers: Optional[List[Optional[LRScheduler]]] = None
    self.run_name = ""
    self.device: Optional[str] = None
    self.device_type: Optional[str] = None
    self.max_len: Optional[int] = None
    self.sections: dict = {}
    self._construct_sections()

  def _construct_sections(self) -> dict:
    """Construct Section wrappers for standard saveable components.

    This method sets up Section objects for the standard components in order:
    - runner: Runner-level hyperparameters (batch_size, etc)
    - model: The neural network model
    - optimizer: The optimizer(s)
    - lr_scheduler: The learning rate scheduler(s)
    - curriculum: The training curriculum/plan

    Child classes can override this method to add custom sections or modify
    existing ones. For example:

      def _construct_sections(self):
        super()._construct_sections()
        self.sections['custom'] = Section('custom',
                                         archive=self.archive_custom,
                                         advertise=self.advertise_custom,
                                         construct=self.construct_custom)

    Returns:
      dict: Mapping of section names to Section objects (preserves order)
    """
    self.sections = {
      'curriculum': Section('curriculum',
                            archive=self.archive_curriculum,
                            advertise=self.advertise_curriculum,
                            construct=self.construct_curriculum),
      'runner': Section('runner',
                        archive=self.archive_runner,
                        advertise=self.advertise_runner,
                        construct=self.construct_runner),
      'model': Section('model',
                       archive=self.archive_model,
                       advertise=self.advertise_model,
                       construct=self.construct_model),
      'optimizer': Section('optimizer',
                           archive=self.archive_optimizer,
                           advertise=self.advertise_optimizer,
                           construct=self.construct_optimizer),
      'lr_scheduler': Section('lr_scheduler',
                              archive=self.archive_lr_scheduler,
                              advertise=self.advertise_lr_scheduler,
                              construct=self.construct_lr_scheduler),
    }
    return self.sections

  def section(self, name: str) -> Section:
    """Get a Section by name, returning a nop Section if not found.

    Args:
      name: Section name (e.g., 'model', 'optimizer')

    Returns:
      Section: The requested section, or a nop Section if not found
    """
    return self.sections.get(name, Section(name))

  def archive_runner(self) -> dict:
    """Archive runner-level hyperparameters and training state.

    Saves runner configuration (batch_size, validation_batch_size, grad_clip, include_prompts,
    validation_interval, save_interval) and training state (statistics).

    Returns:
      dict with 'state_args' (runner config) and 'state' (training state)
    """
    return {
      'state_args': {
        'batch_size': self.batch_size,
        'validation_batch_size': self.validation_batch_size,
        'grad_clip': self.grad_clip,
        'include_prompts': self.include_prompts,
        'validation_interval': self.validation_interval,
        'save_interval': self.save_interval,
      },
      'state': {
        'stats': self.model_stats.dump_dict(),
        'step_stats': {name: stat.dump_dict() for name, stat in self.step_stats.items()},
      }
    }

  def advertise_runner(self) -> dict:
    """Advertise runner construction parameters.

    Returns the current runner configuration so it can be exposed in config files.

    Returns:
      dict of runner configuration parameters
    """
    return {
      'batch_size': self.batch_size,
      'validation_batch_size': self.validation_batch_size,
      'validation_interval': self.validation_interval,
      'save_interval': self.save_interval,
      'default_freeze': False,
    }

  def construct_runner(self, construction_args: dict, state_args: dict, state: dict):
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
      run_name = '_'.join(get_step_names(self.training_plan))

    self.run_name = f"_{run_name}" if len(run_name) > 0 else ""

    # Initialize step tracking
    self.step_stats = dict()

    # Get reset_history from construction_args
    reset_history = construction_args.get('reset_history', False) if construction_args else False

    # Initialize or restore model statistics
    if reset_history or not state or 'stats' not in state:
      # Fresh stats (no checkpoint or reset requested)
      self.model_stats = modelstats.ModelStats(
        model_name=f"model{self.run_name}",
        basedir=self._stats_dir)
    else:
      # Restore stats from checkpoint
      self.model_stats = modelstats.ModelStats(basedir=self._stats_dir, **state['stats'])

    # Restore step-level statistics from checkpoint (unless reset_history)
    if state and not reset_history:
      if 'step_stats' in state:
        for step_name, step_data in state['step_stats'].items():
          self.step_stats[step_name] = modelstats.ModelStats(basedir=self._stats_dir, **step_data)

    # Hyperparameters: state_args override construction_args if provided
    for param_name in ['batch_size', 'validation_batch_size', 'grad_clip', 'include_prompts', 'validation_interval', 'save_interval', 'default_freeze']:
      if state_args and param_name in state_args:
        setattr(self, param_name, state_args[param_name])
      elif construction_args and param_name in construction_args:
        setattr(self, param_name, construction_args[param_name])

  def archive_model(self) -> dict:
    """Archive model state.

    Saves the model's construction arguments (state_args) and weights (state).
    state_args represents the parameters that were used to construct the model,
    while state contains the learned weights and internal tensors.

    Returns:
      dict with 'state_args' (construction args) and 'state' (model weights)
    """
    return {
      'state_args': self._model_args if self._model_args else {},
      'state': self._raw_model.state_dict()
    }

  def advertise_model(self) -> dict:
    """Advertise model construction parameters.

    Asks the model what construction parameters it supports, so they can
    be exposed in config files. This calls the model's advertise_params() method.

    Returns:
      dict of supported construction parameters
    """
    return self._model.advertise_params()

  def construct_model(self, construction_args: dict, state_args: dict, state: dict):
    """Construct model from checkpoint or fresh.

    Sets internal state: self._model, self._raw_model, self._model_args, self.max_len

    Args:
      construction_args: Model construction parameters (num_blocks, etc.) from config defaults
      state_args: Saved model arguments from checkpoint (overrides construction_args)
      state: Model weights (state_dict from checkpoint, empty dict if no checkpoint)
    """
    # Merge: state_args (saved) override construction_args (defaults)
    merged_args = {}
    if construction_args:
      merged_args.update(construction_args)
    if state_args:
      merged_args.update(state_args)

    # Construct model with merged arguments
    # _construct_model returns either (model, args) or (model, args, missing, unexpected)
    if state:
      # Loading from checkpoint with weights
      result = self._construct_model(
        self.device,
        model_weights=state,
        model_args=merged_args,
        strict=False,
        **merged_args
      )
      # Unpack result - with weights we may get 4 values
      if isinstance(result, tuple) and len(result) == 4:
        self._model, self._model_args, missing, unexpected = result
      else:
        self._model, self._model_args = result[0], result[1]
    else:
      # Fresh model without weights
      result = self._construct_model(self.device, **merged_args)
      if isinstance(result, tuple) and len(result) >= 2:
        self._model, self._model_args = result[0], result[1]
      else:
        self._model = result
        self._model_args = merged_args

    self._raw_model = self._model
    self.max_len = self._model.max_len

  def archive_optimizer(self) -> dict:
    """Archive optimizer state.

    Saves the optimizer configuration (state_args) and state. Always returns
    optimizer states as a list for consistency.

    Returns:
      dict with 'state_args' (optimizer config) and 'state' (list of optimizer states)
    """
    optimizer_states = [optimizer.state_dict() for optimizer in self._optimizers]

    return {
      'state_args': self._optimizer_args if self._optimizer_args else {},
      'state': optimizer_states
    }

  def advertise_optimizer(self) -> dict:
    """Advertise optimizer construction parameters.

    Returns the current optimizer configuration (type, lr, weight_decay, etc.)
    so it can be exposed in config files.

    Returns:
      dict of optimizer configuration parameters
    """
    return self._optimizer_args if self._optimizer_args else {}

  def construct_optimizer(self, construction_args: dict, state_args: dict, state: dict):
    """Construct optimizer from checkpoint or fresh.

    Sets internal state: self._optimizers, self._optimizer_args

    One-time parameters (NOT saved to state):
    - load_optimizer: Whether to load saved optimizer state (runtime control only)
    - missing: Missing keys from checkpoint (indicates model incompatibility)
    - unexpected: Unexpected keys from checkpoint (indicates model incompatibility)

    Args:
      construction_args: Optimizer construction parameters (optimizer_type, lr, weight_decay) from config defaults
      state_args: Saved optimizer arguments from checkpoint (overrides construction_args) - clean, no runtime params
      state: Optimizer state_dicts (list of saved optimizer states, empty list if no checkpoint)
    """
    # Must have model already constructed
    if not self._model:
      raise RuntimeError("Model must be constructed before optimizers")

    # Merge: state_args (saved) override construction_args (defaults)
    merged_args = {}
    if construction_args:
      merged_args.update(construction_args)
    if state_args:
      merged_args.update(state_args)

    # Extract one-time runtime parameters (these control behavior but won't be saved)
    load_optimizer = merged_args.pop('load_optimizer', True)
    missing = merged_args.pop('missing', None)
    unexpected = merged_args.pop('unexpected', None)

    # Now merged_args contains only saveable parameters
    self._optimizer_args = merged_args.copy()

    # Get optimizer configuration
    lr = merged_args.get('lr', DEFAULT_LR)
    optimizer_type = merged_args.get('optimizer', 'adamw').lower()
    weight_decay = merged_args.get('weight_decay', get_default_weight_decay(optimizer_type))

    # Categorize parameters for weight decay
    decay_params, no_decay_params = categorize_parameters_by_weight_decay(self._model)
    param_groups = []

    if weight_decay > 0 and len(no_decay_params) > 0:
      if len(decay_params) > 0:
        param_groups.append({'params': decay_params, 'weight_decay': weight_decay})
      param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
    else:
      all_params = decay_params + no_decay_params if weight_decay > 0 else decay_params
      if not all_params:
        all_params = list(self._model.parameters())
      param_groups.append({'params': all_params, 'weight_decay': weight_decay})

    # Create optimizer
    optimizer = create_optimizer(optimizer_type, param_groups, lr, weight_decay)
    self._optimizers = [optimizer]

    # Load optimizer state if available and allowed
    optimizer_weights = state if (load_optimizer and state) else None
    if optimizer_weights:
      if not isinstance(optimizer_weights, list):
        optimizer_weights = [optimizer_weights]

      # Only load if model didn't change (no missing/unexpected keys) and counts match
      if not missing and not unexpected and len(optimizer_weights) == len(self._optimizers):
        for opt_idx, (optimizer, weights) in enumerate(zip(self._optimizers, optimizer_weights)):
          # Check if parameter group structure matches to avoid loading issues
          saved_num_groups = len(weights.get('param_groups', []))
          current_num_groups = len(optimizer.param_groups)

          if saved_num_groups != current_num_groups:
            # Parameter group structure has changed (e.g., new weight decay grouping)
            logger.warning(
              f"Optimizer {opt_idx}: Parameter group structure changed: saved={saved_num_groups}, current={current_num_groups}. "
              f"Skipping optimizer state loading to avoid mismatch."
            )
            continue

          # Update LR and weight decay in saved state to match current config
          first_group = True
          for pg in weights['param_groups']:
            if 'lr' in pg:
              pg['lr'] = lr
            #weight decay is always 0 for second group. This should be bias/other params that weight decay hurts.
            if first_group and 'weight_decay' in pg:
              pg['weight_decay'] = weight_decay
            first_group = False

          try:
            optimizer.load_state_dict(weights)
            logger.info(f"Optimizer {opt_idx}: Loaded state from checkpoint")
          except (ValueError, RuntimeError) as e:
            logger.error(
              f"Optimizer {opt_idx}: Unable to load optimizer state: {e}. Probably a new parameter or structure change. "
              f"Starting with fresh optimizer.")

    # Log optimizer configuration summary
    total_params = sum(p.numel() for p in self._model.parameters())
    logger.info(
      f"Optimizer setup: {len(self._optimizers)} instance(s) of {optimizer_type.upper()}, "
      f"total parameters: {total_params:,}, "
      f"lr={lr}, weight_decay={weight_decay}"
    )

    # Store optimizer type in optimizer_args so it persists in checkpoints
    self._optimizer_args['optimizer'] = optimizer_type
    self._optimizer_args['lr'] = lr
    self._optimizer_args['weight_decay'] = weight_decay

    # Log detailed parameter group information (INFO level for summary, DEBUG for individual params)
    for opt_idx, optimizer in enumerate(self._optimizers):
      for group_idx, param_group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in param_group['params'])
        decay_val = param_group.get('weight_decay', weight_decay)
        logger.info(
          f"  Optimizer {opt_idx}, Group {group_idx}: "
          f"{num_params:,} parameters, weight_decay={decay_val}"
        )
        if logger.isEnabledFor(logging.DEBUG):
          # Use object identity (id) instead of tensor comparison to avoid shape mismatch errors
          param_ids_in_group = {id(p) for p in param_group['params']}
          param_names = [name for name, p in self._model.named_parameters() if id(p) in param_ids_in_group]
          for name in param_names:  # Log all parameter names
            logger.debug(f"    - {name}")

  def archive_lr_scheduler(self) -> dict:
    """Archive learning rate scheduler state.

    Saves the scheduler configuration (state_args) and state. Always returns
    scheduler states as a list for consistency with optimizer handling.

    Returns:
      dict with 'state_args' (scheduler config) and 'state' (list of scheduler states)
    """
    scheduler_states = [scheduler.state_dict() for scheduler in self._lr_schedulers]

    return {
      'state_args': {},  # Schedulers don't have configurable state_args currently
      'state': scheduler_states
    }

  def advertise_lr_scheduler(self) -> dict:
    """Advertise learning rate scheduler construction parameters.

    Returns warmup scheduler parameters. Warmup is a one-time special case that occurs
    when optimizer state is missing/incompatible, allowing LR to ramp up gradually.

    Returns:
      dict with warmup parameters (warmup disabled by default)
    """
    return {
      'optimizer_warmup_fraction': None,  # Initial LR as fraction of target (e.g., 0.1)
      'optimizer_warmup_steps': None,     # Number of steps to warm up over
      'disable_optimizer_warmup': True,   # Warmup disabled by default
    }

  def construct_lr_scheduler(self, construction_args: dict, state_args: dict, state: dict):
    """Construct learning rate scheduler(s) for optimizer(s).

    Creates one scheduler per optimizer in self._optimizers. If warmup is configured and not disabled,
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
    if not self._optimizers:
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
    for optimizer in self._optimizers:
      if create_warmup:
        # Warmup scheduler for this optimizer
        lr_schedulers.append(OptimizerWarmupLRScheduler(
          optimizer,
          steps=optimizer_warmup_steps,
          initial_fraction=optimizer_warmup_fraction))
      else:
        # No-op scheduler
        lr_schedulers.append(NoOpLRScheduler(optimizer))

    self._lr_schedulers = lr_schedulers

  def archive_curriculum(self) -> dict:
    """Archive curriculum state.

    Saves the training plan name/dict to state and current_step to state_args.

    Returns:
      dict with 'state_args' (current_step) and 'state' (training plan)
    """
    return {
      'state_args': {
        'current_step': self.current_step,
      },
      'state': {
        'training_plan_name': self.training_plan_name,
        'training_plan': self.training_plan,
      }
    }

  def advertise_curriculum(self) -> dict:
    """Advertise curriculum construction arguments.

    Exposes training_plan (filename/name of plan to load) and override_plan flag.

    Returns:
      dict with training plan configuration
    """
    return {
      'training_plan': 'default',  # Name or path of training plan (e.g., 'default', 'full', or '/path/to/plan.json')
      'override_plan': False,      # Set to True to load a new plan and discard saved state
    }

  def construct_curriculum(self, construction_args: dict, state_args: dict, state: dict):
    """Construct curriculum by loading training plan and setting current step.

    Loads the training plan from construction_args['training_plan'] if:
    - override_plan=True, or
    - No checkpoint state exists

    Sets current_step to the first step of the training plan, then overrides
    with state_args['current_step'] if provided (allows manual step override).

    Args:
      construction_args: Curriculum construction arguments (training_plan name, override_plan flag)
      state_args: Saved curriculum state arguments (current_step to override)
      state: Saved curriculum state (training_plan_name and training_plan from checkpoint)
    """
    override_plan = construction_args.get('override_plan', False) if construction_args else False
    training_plan_name = construction_args.get('training_plan') if construction_args else 'default'

    # Determine if we should load fresh from construction_args or use checkpoint state
    should_load_fresh = override_plan or not state or 'training_plan' not in state

    if should_load_fresh and training_plan_name:
      # Load training plan from file or DEFAULT_PLANS
      if training_plan_name in DEFAULT_PLANS:
        self.training_plan = DEFAULT_PLANS[training_plan_name]
      else:
        # Try to load from file
        with open(os.path.expanduser(training_plan_name)) as infile:
          self.training_plan = json.loads(infile.read())
      self.training_plan_name = training_plan_name
    else:
      # Use checkpoint state
      self.training_plan_name = state.get('training_plan_name') if state else None
      self.training_plan = state.get('training_plan', {}) if state else {}

    # Set current_step: start with first step from plan, then override if provided
    first_step = get_step_names(self.training_plan)[0] if self.training_plan else None
    self.current_step = first_step

    # Override with saved current_step if provided in state_args
    if state_args and 'current_step' in state_args:
      self.current_step = state_args['current_step']

  def get_construction_args_for_config(self) -> dict:
    """Get construction_args from all sections for config dumping.

    Collects advertised parameters from each section. Useful for dumping
    the current configuration to a config file.

    Returns:
      dict: Section name -> construction args dict
    """
    construction_args = {}
    for section_name in self.sections:
      section = self.section(section_name)
      construction_args[section_name] = section.advertise()
    return construction_args

  def get_state_args_for_config(self) -> dict:
    """Get state_args from all sections for config dumping.

    Collects current state_args from each section. Useful for dumping
    example overrides to a config file (typically commented out).

    Returns:
      dict: Section name -> state_args dict
    """
    state_args = {}
    for section_name in self.sections:
      section = self.section(section_name)
      archived = section.archive()
      state_args[section_name] = archived.get('state_args', {})
    return state_args

  def set_current_step(self, step_name: str):
    """Set the current training step/stage name.

    Changes the active step for statistics tracking. Writes out stats
    for the previous step if switching to a new one.

    Args:
      step_name: Name of the new step
    """
    if not self.current_step is None and step_name != self.current_step:
      self.get_step_stats().write_train()
      self.get_step_stats().write_validate()
    self.current_step = step_name

  def is_trainable(self) -> bool:
    """Check if the runner is configured for training.

    Returns:
      bool: True if optimizers are initialized
    """
    return self._optimizers is not None

  def is_initialized(self) -> bool:
    """Check if the runner has been initialized with a model.

    Returns:
      bool: True if model is loaded
    """
    return self._model is not None

  def get_step_stats(self) -> modelstats.ModelStats:
    """Get or create statistics tracker for current step.

    Returns:
      ModelStats: Statistics tracker for the current step
    """
    if self.current_step not in self.step_stats:
      self.step_stats[self.current_step] = modelstats.ModelStats(
        model_name=f"{self._model.name}{self.run_name}_step_{self.current_step}",
        basedir=self._stats_dir)
    return self.step_stats[self.current_step]

  def initialize(self,
                 construction_args: Dict[str, Any],
                 state_args_overrides: Dict[str, Any],
                 device,
                 locations: Optional[Union[Sequence[str], str]] = None,
                 for_train=True,
                 compile_model=False,
                 compile_mode=None,
                 compile_backend='inductor',
                 amp=False,
                 no_grad_scale=False,
                 check_grads=False,
                 **parameters):
    """Initialize runner using Section architecture.

    Each section (runner, model, optimizer, lr_scheduler, curriculum) receives:
    - construction_args: Configuration parameters from config files
    - state_args: Saved parameters from checkpoint (overrides construction_args)
    - state: Saved internal state from checkpoint

    Args:
      construction_args: Dict[section_name -> args] from config files
      state_args_overrides: Dict[section_name -> args] from command line overrides
      device: Device to run on (e.g., 'cuda', 'cpu')
      locations: Optional checkpoint file path(s) to load from
      for_train: Whether to initialize for training (vs inference only)
      compile_model: Whether to compile model with torch.compile
      compile_mode: Compilation mode (e.g., 'default', 'reduce-overhead')
      compile_backend: Compilation backend (e.g., 'inductor')
      amp: Enable automatic mixed precision
      no_grad_scale: Disable gradient scaling with AMP
      check_grads: Print parameters without gradients
      **parameters: Additional model-specific parameters
    """
    # Set runtime infrastructure variables (not saved, runtime only)
    self.device = device
    self.device_type = ("cuda" if "cuda" in device else
                       "mps" if "mps" in device else
                       "cpu" if "cpu" in device else
                       device)
    self.compile_model = compile_model
    self.compile_mode = compile_mode
    self.compile_backend = compile_backend
    self.check_grads = check_grads
    self.for_train = for_train

    # Initialize global torch settings
    torch.set_float32_matmul_precision('high')

    # Clean up None parameters to avoid confusion
    for p in ('lr', 'optimizer_warmup_start', 'optimizer_warmup_steps', 'max_len'):
      if p in parameters and parameters[p] is None:
        del parameters[p]

    # Process and validate checkpoint locations
    if locations is None:
      locations = []
    if isinstance(locations, str):
      locations = [locations]
    locations = [os.path.expanduser(loc) for loc in locations if os.path.exists(os.path.expanduser(loc))]

    # Load checkpoint if available
    checkpoint = {}
    if len(locations) > 0:
      location = locations[0]
      checkpoint = torch.load(location, map_location=device, weights_only=False)

    # Extract section data from checkpoint
    checkpoint_sections = {}
    for section_name in self.sections:
      if section_name in checkpoint:
        checkpoint_sections[section_name] = checkpoint[section_name]

    # Merge state_args: checkpoint state_args + command-line overrides
    merged_state_args = {}
    for section_name in self.sections:
      merged_state_args[section_name] = {}
      # Start with checkpoint state_args if present
      if section_name in checkpoint_sections and 'state_args' in checkpoint_sections[section_name]:
        merged_state_args[section_name].update(checkpoint_sections[section_name]['state_args'])
      # Apply command-line overrides
      if state_args_overrides and section_name in state_args_overrides:
        merged_state_args[section_name].update(state_args_overrides[section_name])

    # Prepare state dicts for each section
    checkpoint_state = {}
    for section_name in self.sections:
      if section_name in checkpoint_sections and 'state' in checkpoint_sections[section_name]:
        checkpoint_state[section_name] = checkpoint_sections[section_name]['state']
      else:
        checkpoint_state[section_name] = {}

    # Build construction_args per section
    section_construction_args = {}
    for section_name in self.sections:
      section_construction_args[section_name] = {}
      # Start with section-specific construction args from config
      if construction_args and section_name in construction_args:
        section_construction_args[section_name].update(construction_args[section_name])
      # Add general parameters (model-specific, optimizer params, etc.)
      section_construction_args[section_name].update(parameters)

    # Call construct() on each section in order
    for section_name in self.sections:
      section = self.section(section_name)
      section.construct(
        section_construction_args.get(section_name, {}),
        merged_state_args.get(section_name, {}),
        checkpoint_state.get(section_name, {})
      )

    # Post-construction setup
    # Model must be constructed by now
    self.max_len = self._model.max_len

    # Handle parameter freezing for training
    if for_train:
      # Apply default freeze if configured
      if self.default_freeze:
        logger.info("Applying default_freeze: freezing all model parameters")
        self._model.requires_grad_(False)
      detect_freeze(self._model)

    # Apply torch.compile if requested
    if compile_model:
      # Set sensible defaults for compilation
      if self.compile_backend is None:
        self.compile_backend = 'inductor' if 'cuda' in device else 'eager'
      if self.compile_mode is None:
        self.compile_mode = 'reduce-overhead'

      print(f"Compiling model using {self.compile_backend}:{self.compile_mode}")
      self._model = torch.compile(
        self._model,
        backend=self.compile_backend,
        mode=self.compile_mode,
        fullgraph=False,
        disable=False,
      )

    # Set up AMP infrastructure
    self.scaler = None
    self.amp = NopWith
    if amp:
      if "cuda" in device and not no_grad_scale:
        self.scaler = torch.amp.GradScaler('cuda')
      self.amp = torch.amp.autocast

    # Set model to train or eval mode
    self._model.train(for_train)

  def get_model_args(self):
    """Get the model initialization arguments.

    Returns:
      Model initialization arguments dictionary
    """
    return self._model_args

  def get_optimizer_args(self):
    """Get the optimizer initialization arguments.

    Returns:
      Optimizer initialization arguments dictionary
    """
    return self._optimizer_args

  def save(self, location: str, prod_save=False):
    """Save model checkpoint to disk using Section architecture.

    All sections are archived in a uniform structure. For prod_save, only saves
    model weights and args without optimizer/stats.

    Args:
      location: Path to save the checkpoint
      prod_save: If True, only save model section (no optimizer/stats/runner)
    """
    assert self.is_initialized(), "Runner not initialized"

    checkpoint = {}

    if prod_save:
      # Only save model for production use
      checkpoint['model'] = self.section('model').archive()
    else:
      # Save all sections uniformly
      for section_name in self.sections:
        checkpoint[section_name] = self.section(section_name).archive()

    if os.path.exists(location):
      copyfile(location, f"{location}.bak")
    torch.save(checkpoint, location)

  def _calculate_stats(self, prompts_data: Sequence[dict], results: Sequence[str]):
    """Calculate accuracy statistics for predictions.

    Args:
      prompts_data: Original prompt data with ground truth
      results: Model predictions

    Returns:
      tuple: (total_words, total_errors, total_matches)
    """
    total_words = 0
    total_errors = 0
    total_matches = 0
    for result, prompt_data in zip(results, prompts_data):
      truth = prompt_data['truth']
      result = result.split()
      truth = truth.split()
      total_words += len(truth)
      # The wrong library here can make this the most expensive op in the codebase.
      # This lib is pretty fast though.
      errors, matches = utils.levenshtein_edit_distance(result, truth)

      total_errors += errors
      total_matches += matches
    return total_words, total_errors, total_matches

  def _run_with_truth(self,
                      prompts: Sequence[dict],
                      train: bool,
                      actual_samples_read: int) -> (Sequence[str], torch.Tensor):
    """Run model on prompts with ground truth, handling batching and gradients.

    Args:
      prompts: Sequence of prompt dictionaries
      train: Whether to compute and accumulate gradients
      actual_samples_read: Actual number of samples (for statistics)

    Returns:
      tuple: (predictions, loss, token_count)
    """
    # This will batch to max batch size and pass to the model then re-package the results to return the result.
    # If the passed in batch is more than max_batch_size then gradient accumulation will be used.
    # Tokenization is not done here because the model is the only thing that knows how to do all that.
    assert self.is_initialized(), "Runner not initialized"
    assert self.is_trainable(), "Runner not trainable"
    mini_batch = []
    batch_results = []
    batch_loss = 0.0
    total_tokens = 0
    # Break this into mini-batches that the model can handle
    # AMP autocast should only wrap the forward pass, not backward.
    # Backward pass must occur outside autocast for proper numerical stability.
    for prompt in prompts:
      mini_batch.append(prompt)
      if len(mini_batch) >= self.max_batch_size:
        # Forward pass inside autocast
        with self.amp(device_type=self.device_type):
          mini_batch_results, mini_batch_loss, mini_batch_token_count = self._model.train_prompts(mini_batch,
                                                                                                  include_prompts=self.include_prompts)
        batch_results.extend(mini_batch_results)
        batch_loss = float(mini_batch_loss.item()) + batch_loss
        total_tokens += mini_batch_token_count
        # Loss computation and backward outside autocast
        mini_batch_fraction = len(mini_batch) / len(prompts)
        mini_batch_loss = (mini_batch_loss * mini_batch_fraction) / mini_batch_token_count
        if train:
          # accumulate the gradients - outside autocast for stability
          if self.scaler is not None:
            self.scaler.scale(mini_batch_loss).backward()
          else:
            mini_batch_loss.backward()

        mini_batch = []
    if len(mini_batch) > 0:
      # Forward pass inside autocast
      with self.amp(device_type=self.device_type):
        mini_batch_results, mini_batch_loss, mini_batch_token_count = self._model.train_prompts(mini_batch,
                                                                                                include_prompts=self.include_prompts)
      total_tokens += mini_batch_token_count
      batch_results.extend(mini_batch_results)
      batch_loss = float(mini_batch_loss.item()) + batch_loss
      # Loss computation and backward outside autocast
      mini_batch_fraction = len(mini_batch) / len(prompts)
      mini_batch_loss = (mini_batch_loss * mini_batch_fraction) / mini_batch_token_count

      if train:
        # accumulate the gradients - outside autocast for stability
        if self.scaler is not None:
          self.scaler.scale(mini_batch_loss).backward()
        else:
          mini_batch_loss.backward()

    # normalize on total tokens.
    batch_loss = batch_loss / total_tokens

    # Get basic accuracy stats so we can update the training stats
    tw, te, tm = self._calculate_stats(prompts, batch_results)
    if tw > 0:
      pct_correct = tm / tw
    elif te > 0:
      pct_correct = 0
    else:
      pct_correct = 0
    if train:
      self.model_stats.update_train(total_tokens,
                                    len(prompts),
                                    pct_correct,
                                    float(batch_loss),
                                    actual_samples=actual_samples_read)
      self.get_step_stats().update_train(total_tokens,
                                         len(prompts),
                                         pct_correct,
                                         float(batch_loss),
                                         actual_samples=actual_samples_read)
    else:
      self.model_stats.update_validate(total_tokens,
                                       len(prompts),
                                       pct_correct,
                                       float(batch_loss),
                                       actual_samples=actual_samples_read)
      self.get_step_stats().update_validate(total_tokens,
                                            len(prompts),
                                            pct_correct,
                                            float(batch_loss),
                                            actual_samples=actual_samples_read)
    return batch_results, batch_loss, total_tokens

  def train(self, prompts: Sequence[dict], actual_samples_read: Optional[int] = None) -> (
  Sequence[str], torch.Tensor, int):
    """Execute a training step on the given prompts.

    Args:
      prompts: Training samples with prompts and ground truth
      actual_samples_read: Actual samples read (for accurate statistics)

    Returns:
      tuple: (predictions, loss, total_tokens)
    """
    # The assumption is they have sent in a whole batch that they want loss accumulated over
    # But the model may not support the size they send to us.
    # So we will break it into mini batches and do gradient accumulation.
    torch.compiler.cudagraph_mark_step_begin()
    if not actual_samples_read:
      actual_samples_read = len(prompts)
    for optimizer in self._optimizers:
      optimizer.zero_grad()

    results, current_loss, total_tokens = self._run_with_truth(prompts, True, actual_samples_read)
    if not self.grad_clip is None:
      if not self.scaler is None:
        self.scaler.unscale_(self._optimizers[0])
      torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)

    if self.check_grads:
      for name, param in self._model.named_parameters():
        if param.grad is None:
          logger.warning(f"No gradient found for parameter: {name}")

    if self.scaler is not None:
      # Scaling only applies to the primary optimizer.
      self.scaler.step(self._optimizers[0])
      self.scaler.update()
    else:
      self._optimizers[0].step()
    for optimizer in self._optimizers[1:]:
      optimizer.step()
    # Step all schedulers (one per optimizer)
    for scheduler in self._lr_schedulers:
      scheduler.step()
    return results, current_loss, total_tokens

  def validate(self, prompts: Sequence[dict], actual_samples_read: Optional[int] = None) -> (
          Sequence[str], torch.Tensor, int):
    """Execute a validation step on the given prompts.

    Args:
      prompts: Validation samples with prompts and ground truth
      actual_samples_read: Actual samples read (for accurate statistics)

    Returns:
      tuple: (predictions, loss, total_tokens)
    """
    self._model.train(False)
    if not actual_samples_read:
      actual_samples_read = len(prompts)
    results, current_loss, total_tokens = self._run_with_truth(prompts, False, actual_samples_read)
    self._model.train(True)
    return results, current_loss, total_tokens

  def generate(self,
               prompts: Sequence[str],
               max_len: Optional[int] = None,
               temperature: float = 1.0,
               top_k: Optional[int] = None,
               top_p: float = 1.0,
               repetition_penalty: float = 1.0,
               do_sample: bool = False):
    """Generate text completions for prompts.

    Args:
      prompts: Text prompts to complete
      max_len: Maximum generation length
      temperature: Sampling temperature (< 1.0 = sharper, > 1.0 = softer)
      top_k: Keep only top k tokens. None/0 = disabled.
      top_p: Nucleus sampling threshold. 1.0 = disabled.
      repetition_penalty: Penalty for repeated tokens. 1.0 = no penalty.
      do_sample: Use sampling instead of greedy decoding.

    Returns:
      List of generated completions
    """
    prompts = [{'prompt': f"{prompt}\n"} for prompt in prompts]
    with self.amp(device_type=self.device_type):
      return self._model.generate_prompts(
          prompts,
          max_len=max_len,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          repetition_penalty=repetition_penalty,
          do_sample=do_sample)

  def run(self, prompts: Sequence[dict]) -> Sequence[str]:
    """Run inference on prompts without ground truth.

    Args:
      prompts: Sequence of prompt dictionaries

    Returns:
      List of model outputs
    """
    # This will batch to max batch size and pass to _run then re-package the results to return the result
    # Tokenization is not done here because the model is the only thing that knows how to do all that.
    batch = []
    results = []
    with torch.no_grad():
      for prompt in prompts:
        batch.append(prompt)
        if len(batch) >= self.max_batch_size:
          results.extend(self._model(batch))
          batch = []
      if len(batch) > 0:
        results.extend(self._model(batch))
    return results

  @abstractmethod
  def _construct_model(self, device, model_weights: dict = None, model_args=None, strict=False, **parameters) -> (
          LMBase, Any):
    """Construct the model instance. Must be implemented by subclasses.

    Args:
      device: Device to place model on
      model_weights: Optional pre-trained weights
      model_args: Optional saved model arguments
      strict: Whether to enforce strict weight loading
      **parameters: Additional parameters

    Returns:
      tuple: (model instance, model arguments)
    """
    pass

