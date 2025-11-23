import logging
import os
from abc import ABC, abstractmethod
from shutil import copyfile
from typing import Optional, List, Dict, Any, Union, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lmplay.base.base_model import LMBase
from lmplay.base.runner.common import Component, NOPComponent
from lmplay.base.utils import NopWith
from lmplay.stats import modelstats, utils
from lmplay.train.datasets.plan_configs import DEFAULT_PLANS
logger = logging.getLogger(__name__)


class LMRunnerCommon(ABC):
  """Abstract base class for model runners that handle training and inference.

  Runners wrap models and provide high-level functionality for:
  - Model initialization and device management
  - Training loops with gradient accumulation
  - Validation and generation
  - Checkpoint saving/loading
  - Statistics tracking
  - Optimizer and scheduler management

  Attributes:
    mini_batch_size: Maximum batch size for gradient accumulation
    stats_dir: Directory for saving statistics files
  """

  def __init__(self, mini_batch_size: int = 1, stats_dir="./out_gpt"):
    self.mini_batch_size = mini_batch_size
    self.stats_dir = stats_dir

    #Set up reasonable defaults
    #This should be overridden by the curriculum
    self.training_plan_name: str = "default"
    self.training_plan = DEFAULT_PLANS[self.training_plan_name]


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
    self._lr_schedulers: Optional[List[Optional[LRScheduler]]] = None
    self.run_name = ""
    self.device: Optional[str] = None
    self.device_type: Optional[str] = None
    self.max_len: Optional[int] = None
    self.components: dict[str, Component] = dict()
    self._constructed_components: set[str] = set()
    self._add_components(self.components)

  @abstractmethod
  def _add_components(self, existing_components:dict[str, Component]):
    pass


  def component(self, name: str) -> Component:
    """Get a component by name, returning a NOPComponent if not found.

    Args:
      name: component name (e.g., 'model', 'optimizer')

    Returns:
      Component: The requested component, or a NOPComponent if not found
    """
    if name not in self.components:
      logger.warning(f"Didn't find component {name}. Creating a nop component that ignores these parameters.")
      self.components[name] = NOPComponent()
      self._constructed_components.add(name)
    return self.components[name]




  def get_construction_args_for_config(self) -> dict:
    """Get construction_args from all components for config dumping.

    Collects advertised parameters from each component. Useful for dumping
    the current configuration to a config file.

    Returns:
      dict: component name -> construction args dict
    """
    construction_args = {}
    for component_name in self.components:
      component = self.component(component_name)
      construction_args[component_name] = component.advertise()
    return construction_args

  def get_state_args_for_config(self) -> dict:
    """Get state_args from all component for config dumping.

    Collects current state_args from each component. Useful for dumping
    example overrides to a config file (typically commented out).

    Returns:
      dict: component name -> state_args dict
    """
    state_args = {}
    for component in self.components:
      component = self.component(component)
      archived = component.archive()
      state_args[component] = archived.get('state_args', {})
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
              basedir=self.stats_dir)
    return self.step_stats[self.current_step]

  def _construct_component(self,
                           component_name: str,
                           construction_args: dict,
                           state_args: dict,
                           state: dict):
    assert component_name not in self._constructed_components, f"{component_name} already constructed"
    self.component(component_name).construct(construction_args,
                                           state_args,
                                           state)
    self._constructed_components.add(component_name)

  def _construct_remaining_components(self,
                                      construction_args_by_component: dict,
                                      state_args_by_component: dict,
                                      state_by_component: dict):
    # Call construct() on each component in order
    for component_name in self.components:
      if component_name not in self._constructed_components:
        self._construct_component(component_name,
                                  construction_args_by_component.get(component_name, {}),
                                  state_args_by_component.get(component_name, {}),
                                  state_by_component.get(component_name, {}))

  def _construct_components(self,
                            construction_args_by_component: dict,
                            state_args_by_component: dict,
                            state_by_component: dict):
    self._construct_remaining_components(construction_args_by_component,
                                         state_args_by_component,
                                         state_by_component)

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
                 check_grads=False):
    """Initialize runner using component architecture.

    Each component (runner, model, optimizer, lr_scheduler, curriculum) receives:
    - construction_args: Configuration parameters from config files
    - state_args: Saved parameters from checkpoint (overrides construction_args)
    - state: Saved internal state from checkpoint

    Args:
      construction_args: Dict[component_name -> args] from config files
      state_args_overrides: Dict[component_name -> args] from command line overrides
      device: Device to run on (e.g., 'cuda', 'cpu')
      locations: Optional checkpoint file path(s) to load from
      for_train: Whether to initialize for training (vs inference only)
      compile_model: Whether to compile model with torch.compile
      compile_mode: Compilation mode (e.g., 'default', 'reduce-overhead')
      compile_backend: Compilation backend (e.g., 'inductor')
      amp: Enable automatic mixed precision
      no_grad_scale: Disable gradient scaling with AMP
      check_grads: Print parameters without gradients
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

    # Extract component data from checkpoint
    checkpoint_components = {}
    for component_name in self.components:
      if component_name in checkpoint:
        checkpoint_components[component_name] = checkpoint[component_name]

    # Merge state_args: checkpoint state_args + command-line overrides
    merged_state_args = {}
    for component_name in self.components:
      merged_state_args[component_name] = {}
      # Start with checkpoint state_args if present
      if component_name in checkpoint_components and 'state_args' in checkpoint_components[component_name]:
        merged_state_args[component_name].update(checkpoint_components[component_name]['state_args'])
      # Apply command-line overrides
      if state_args_overrides and component_name in state_args_overrides:
        merged_state_args[component_name].update(state_args_overrides[component_name])

    # Prepare state dicts for each component
    checkpoint_state = {}
    for component_name in self.components:
      if component_name in checkpoint_components and 'state' in checkpoint_components[component_name]:
        checkpoint_state[component_name] = checkpoint_components[component_name]['state']
      else:
        checkpoint_state[component_name] = {}

    # Build construction_args per component
    component_construction_args = {}
    for component_name in self.components:
      component_construction_args[component_name] = {}
      # Start with component-specific construction args from config
      if construction_args and component_name in construction_args:
        component_construction_args[component_name].update(construction_args[component_name])


    self._construct_components(component_construction_args,
                               merged_state_args,
                               checkpoint_state)

    # Post-construction setup
    # Model must be constructed by now
    self.max_len = self._model.max_len


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


  def save(self, location: str, prod_save=False):
    """Save model checkpoint to disk using component architecture.

    All components are archived in a uniform structure. For prod_save, only saves
    model weights and args without optimizer/stats.

    Args:
      location: Path to save the checkpoint
      prod_save: If True, only save model component (no optimizer/stats/runner)
    """
    assert self.is_initialized(), "Runner not initialized"

    checkpoint = {}

    if prod_save:
      # Only save model for production use
      checkpoint['model'] = self.component('model').archive()
    else:
      # Save all components uniformly
      for component_name in self.components:
        checkpoint[component_name] = self.component(component_name).archive()

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
      if len(mini_batch) >= self.mini_batch_size:
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
        if len(batch) >= self.mini_batch_size:
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