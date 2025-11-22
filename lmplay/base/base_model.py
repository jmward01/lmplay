"""Base model classes and training infrastructure for the lmplay framework.

This module provides the core abstractions for building and training neural network models,
particularly language models. It includes:

- MBase: The foundational model class that all models inherit from
- LMBase: Specialized base class for language models
- LMRunnerBase: Abstract base for model runners that handle training/inference
- BasicModelRunner: Simple implementation of a model runner
- OptimizerWarmupLRScheduler: Learning rate scheduler with warmup
- Helper utilities for gradient freezing and context management

The module implements a runner pattern where models are wrapped in runners that
manage the training loop, optimization, checkpointing, and statistics tracking.
"""

import os.path
import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Any, List
import torch
from torch import nn
from lmplay.stats import modelstats, utils
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.utils.rnn import pad_sequence
from shutil import copyfile
import torch.nn.functional as F
from lmplay.utils import ignore_default

logger = logging.getLogger(__name__)

DEFAULT_LR = 1e-4  # Tuned for AdamW (default optimizer). Use 6e-4 with Adagrad
# DEFAULT_LR = 3e-5
DEFAULT_WEIGHT_DECAY = 0.00


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
  """Scale logits by temperature for controlling output diversity.

  Args:
    logits: Model output logits of shape (batch, vocab_size)
    temperature: Temperature value. < 1.0 makes distribution sharper (more deterministic),
                 > 1.0 makes it softer (more random). 1.0 is unchanged.

  Returns:
    Temperature-scaled logits
  """
  if temperature == 1.0:
    return logits
  if temperature <= 0:
    raise ValueError("Temperature must be positive")
  return logits / temperature


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor,
                             penalty: float) -> torch.Tensor:
  """Apply repetition penalty to discourage repeated tokens.

  Args:
    logits: Model output logits of shape (vocab_size,)
    input_ids: Previously generated token IDs
    penalty: Penalty factor. > 1.0 discourages repetition. 1.0 is no penalty.

  Returns:
    Logits with repetition penalty applied
  """
  if penalty == 1.0:
    return logits
  if penalty <= 0:
    raise ValueError("Repetition penalty must be positive")

  # Get unique tokens that have been generated
  unique_tokens = torch.unique(input_ids)
  for token_id in unique_tokens:
    # Reduce logit by dividing (makes tokens less likely)
    logits[token_id] = logits[token_id] / penalty

  return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
  """Filter to keep only top k tokens.

  Args:
    logits: Model output logits of shape (vocab_size,)
    top_k: Number of highest probability tokens to keep. 0 or None disables.

  Returns:
    Logits with non-top-k values set to -inf
  """
  if top_k is None or top_k <= 0:
    return logits

  top_k = min(top_k, logits.size(-1))
  top_k_logits, top_k_indices = torch.topk(logits, top_k)

  # Create mask with -inf for non-top-k tokens
  logits_mask = torch.full_like(logits, float('-inf'))
  logits_mask.scatter_(0, top_k_indices, top_k_logits)

  return logits_mask


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
  """Apply nucleus sampling (top-p filtering).

  Args:
    logits: Model output logits of shape (vocab_size,)
    top_p: Cumulative probability threshold. Tokens with cumulative prob > top_p are filtered.
           1.0 disables. Typical values: 0.9, 0.95

  Returns:
    Logits with tokens outside cumulative probability set to -inf
  """
  if top_p >= 1.0:
    return logits
  if top_p <= 0:
    raise ValueError("top_p must be in (0, 1]")

  # Sort probabilities in descending order
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  sorted_probs = torch.softmax(sorted_logits, dim=-1)

  # Compute cumulative probabilities
  cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

  # Keep tokens until cumulative prob exceeds top_p, but always keep at least one token
  mask = cumsum_probs <= top_p
  mask[0] = True  # Always keep the highest probability token
  sorted_logits[~mask] = float('-inf')

  # Unsort back to original order
  result = torch.full_like(logits, float('-inf'))
  result.scatter_(0, sorted_indices, sorted_logits)

  return result


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


class MBase(nn.Module):
  """Base class for all models in the lmplay framework.
  
  This class provides core functionality for model initialization, tokenization,
  training, generation, and device management. All models should inherit from
  this class or one of its subclasses.
  
  Attributes:
    name: Model name for identification and file saving
    max_len: Maximum sequence length the model can handle
    expect_extra_loss: Whether the model returns additional loss terms
    pass_lengths: Whether to pass sequence lengths to the forward method
    flat_batch: Whether to flatten batches for processing
    tokenizer: Tokenizer instance for text processing
  """
  
  @ignore_default
  def __init__(self,
               name: str,
               *init_args,
               expect_extra_loss=False,
               pass_lengths=False,
               flat_batch=False,
               **init_kwargs):
    super().__init__()
    self.name = name.replace('.', '_')
    self.init_args = init_args
    self.init_kwargs = init_kwargs
    self.max_len = init_kwargs['max_len']
    self.expect_extra_loss = expect_extra_loss
    self.pass_lengths = pass_lengths
    self.flat_batch = flat_batch
    self._cached_device = None

  @property
  def device(self):
    """Get the device of the model, with caching for performance.
    
    Returns:
      torch.device: The device where the model parameters are located
    """
    if self._cached_device is None:
      # Use fc.weight.device as the reference for the model's device
      self._cached_device = self.fc.weight.device
    return self._cached_device

  def initialize(self, device):
    """Initialize model components after moving to device.
    
    This method can be overridden by subclasses to perform device-specific
    initialization.
    
    Args:
      device: The device to initialize on
    """
    pass
    # self.unified_tok_embed.initialize(self.tok_embed, device)

  def _kv_cache(self, cache: Optional[list], idx):
    """Get or create cache entry for key-value caching.
    
    Args:
      cache: Optional list of cached values
      idx: Index for the cache entry
      
    Returns:
      The cache entry at the specified index, or None if cache is None
    """
    if cache is None:
      return None
    if len(cache) <= idx:
      cache.append([])
    return cache[idx]

  def _tokenize_str(self, sample: dict, device, trim=True) -> (torch.Tensor, int):
    """Tokenize a single sample into tensor format.
    
    Args:
      sample: Dictionary with 'prompt' and optionally 'truth' keys
      device: Device to place the tensor on
      trim: Whether to trim sequences that exceed max_len
      
    Returns:
      tuple: (tokens tensor, prediction_starts index)
    """
    prompt = sample['prompt']
    tokens = [self.tokenizer.eot_token] + self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    # adjusting because the the model is - 1 on its prediction.
    # for example if the model sees this (& = bos/eos):
    # & P R E D I C
    # It will predict like this:
    # P R E D I C T
    # So it we said the prompt ended on 1 then prediction starts on 0
    prediction_starts = len(tokens) - 1
    if 'truth' in sample:
      truth: str = sample['truth']
      if not truth.endswith("<|endoftext|>"):
        tokens.extend(self.tokenizer.encode(sample['truth'] + "<|endoftext|>", allowed_special={"<|endoftext|>"}))
      else:
        tokens.extend(self.tokenizer.encode(sample['truth'], allowed_special={"<|endoftext|>"}))

    # We can go one more because one is being trimmed off
    if trim and len(tokens) > self.max_len + 1:
      # too long. Just cut it off
      tokens = tokens[:self.max_len + 1]
      # tokens[-1] = self.tokenizer.eot_token
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    return tokens, prediction_starts

  def _tokenize_batch(self, batch: Sequence[dict], dont_pad=False) -> (torch.Tensor, Sequence[int]):
    """Tokenize a batch of samples and optionally pad them.
    
    Args:
      batch: Sequence of sample dictionaries
      dont_pad: If True, skip padding (used for flat_batch mode)
      
    Returns:
      tuple: (padded tokens tensor, prediction_starts list, prediction_ends list)
    """
    device = self.device  # Use cached device property
    predictions_starts = []
    predictions_ends = []
    x = []
    for t in batch:
      t, ps = self._tokenize_str(t, device)
      x.append(t)
      predictions_starts.append(ps)
      predictions_ends.append(int(t.size(-1)) - 1)
    if not dont_pad:
      x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.eot_token)
    return x, predictions_starts, predictions_ends

  def to(self, *args, **kwargs):
    """Move model to specified device/dtype, with cache invalidation.
    
    Overrides PyTorch's to() method to ensure cached device is cleared
    when the model is moved.
    
    Returns:
      Self for method chaining
    """
    # Modified from pytorch 2.1 source
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    
    # Clear cached device when model is moved
    if device is not None:
      self._cached_device = None

    def convert(t):
      if convert_to_format is not None and t.dim() in (4, 5):
        if hasattr(t, 'force_device'):
          return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None,
                      non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking, memory_format=convert_to_format)
      if hasattr(t, 'force_device'):
        return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
      return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)

  @abstractmethod
  def forward(self, *args, **kwargs):
    """Forward pass of the model. Must be implemented by subclasses.
    
    Returns:
      Model outputs (specific format depends on the model type)
    """
    pass

  def train_prompts(self, prompts: Sequence[dict], include_prompts=True) -> (Sequence[str], torch.Tensor):
    """Process prompts for training, computing loss and predictions.
    
    Args:
      prompts: Sequence of prompt dictionaries with 'prompt' and 'truth' keys
      include_prompts: Whether to include prompt tokens in loss calculation
      
    Returns:
      tuple: (predictions list, total loss, total token count)
    """
    # We want to pad them together so that the truths will line up with the prompts.
    x, predictions_starts, predictions_ends = self._tokenize_batch(prompts, dont_pad=self.flat_batch)
    if self.flat_batch:
      truths = torch.concat([t[1:] for t in x], dim = 0)
      x = torch.concat([t[:-1] for t in x], dim = 0)
    else:
      # Truth doesn't have the first EOT char. It needs to start on prediction start
      truths = x[:, 1:]
      # x doesn't need the last EOT since it will be predicting that
      x = x[:, :-1]

    if self.pass_lengths:
      #These ends should already be one short/match the actual prediction end
      x = self(x, lengths=torch.tensor(predictions_ends, dtype=torch.int64, device=x.device))
    else:
      x = self(x)

    if self.expect_extra_loss:
      x, extra_loss = x
    else:
      extra_loss = None
    results = []
    if self.flat_batch:
      # num classes is always second. For, reasons?
      target_loss = F.cross_entropy(x.unsqueeze(0).permute(0, 2, 1), truths.unsqueeze(0), reduction="none").squeeze(0)

      if not extra_loss is None and extra_loss.shape[0] == target_loss.shape[0]:
        target_loss = target_loss + extra_loss
        extra_loss = None

      target_loss = torch.split(target_loss, predictions_ends)
    else:
      target_loss = F.cross_entropy(x.permute(0, 2, 1), truths, reduction="none")

    # num classes is always second. For, reasons?
    if extra_loss is None:
      total_target_loss = 0.0
    else:
      total_target_loss = torch.sum(extra_loss)
    total_token_count = 0
    for tl, prediction_start, prediction_end in zip(target_loss, predictions_starts, predictions_ends):
      if not include_prompts:
        tl = tl[prediction_start:prediction_end]
      else:
        tl = tl[:prediction_end]
        prediction_start = 0
      tl = tl.sum()
      token_count = max(prediction_end - prediction_start, 1)
      total_token_count += token_count
      # norm by number of tokens in the truth
      # total_target_loss = tl / token_count + total_target_loss
      total_target_loss = total_target_loss + tl

    # Get the predicted value so we can cut just that out as the result.
    predicted_tokens = torch.argmax(x, dim=-1)
    # for result, prediction_start, prediction_end, truth in zip(predicted_tokens, predictions_starts, predictions_ends, truths):
    if self.flat_batch:
      for result in torch.split(predicted_tokens, predictions_ends):
        # we only care about the prediction.
        # the last value is end of sentence
        results.append(self.tokenizer.decode(result.tolist()))
    else:

      for result, prediction_start, prediction_end in zip(predicted_tokens, predictions_starts, predictions_ends):
        # we only care about the prediction.
        # the last value is end of sentence
        results.append(self.tokenizer.decode(result[prediction_start:prediction_end].tolist()))
    # target loss is normalized by example but not by batch. That will be done by the caller.
    return results, total_target_loss, total_token_count

  def generate_prompts(self,
                       prompts: Sequence[dict],
                       max_len: Optional[int] = None,
                       temperature: float = 1.0,
                       top_k: Optional[int] = None,
                       top_p: float = 1.0,
                       repetition_penalty: float = 1.0,
                       do_sample: bool = False) -> Sequence[str]:
    """Generate text completions for given prompts with sampling options.

    Args:
      prompts: Sequence of prompt dictionaries with 'prompt' key
      max_len: Maximum generation length (uses model max_len if None)
      temperature: Sampling temperature. < 1.0 = sharper distribution (more deterministic),
                   > 1.0 = softer distribution (more random). 1.0 = no change.
      top_k: Keep only top k highest probability tokens. None/0 = disabled.
      top_p: Nucleus sampling - keep tokens with cumulative prob <= top_p. 1.0 = disabled.
      repetition_penalty: Penalty for repeated tokens. > 1.0 discourages repetition.
      do_sample: If True, use sampling. If False, use greedy decoding (argmax).

    Returns:
      List of generated text strings
    """
    results = []
    if not max_len:
      max_len = self.max_len

    with torch.no_grad():
      for prompt in prompts:
        result = []
        # We only support batch size of 1. This code is just for testing and not meant to be fast/good/etc.
        # Basically, we want training to be -really- easy to play with so we have the minor concession of a kv cache mechanism.
        x, _, _ = self._tokenize_batch([prompt])
        cache = []
        stop = False

        while not stop:
          x, cache = self(x, cache=cache)
          # x shape: (batch, seq_len, vocab_size) -> take last token and squeeze batch
          logits = x[0, -1, :] if x.dim() == 3 else x[-1, :]

          # Apply generation filters in order
          logits = apply_temperature(logits, temperature)
          logits = apply_repetition_penalty(logits, torch.tensor(result, device=logits.device), repetition_penalty)
          logits = top_k_filtering(logits, top_k)
          logits = top_p_filtering(logits, top_p)

          # Sample or argmax
          if do_sample:
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
          else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

          token_id = next_token.squeeze().item()
          result.append(token_id)

          if token_id == self.tokenizer.eot_token or len(result) == max_len:
            stop = True
          else:
            # Prepare input for next iteration: new token as long tensor
            x = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        results.append(self.tokenizer.decode(result))

    return results

  def parameter_count(self) -> int:
    """Count total number of parameters in the model.
    
    Returns:
      int: Total parameter count
    """
    pc = 0
    for p in self.parameters():
      p_count = 1
      for s in p.shape:
        p_count *= s
      pc += p_count
    return pc


class LMBase(MBase):
  """Base class specifically for language models.
  
  This class extends MBase with a forward signature appropriate for
  autoregressive language modeling with optional key-value caching.
  """
  
  @abstractmethod
  def forward(self, x: torch.Tensor, cache: Optional[List] = None) -> torch.Tensor:
    """Forward pass for language modeling.
    
    Args:
      x: Input token indices of shape (batch_size, sequence_length)
      cache: Optional key-value cache for efficient generation
      
    Returns:
      Output logits and optionally updated cache
    """
    pass


def get_weight_decay_exclusion_patterns() -> Sequence[str]:
  """Get default parameter name patterns to exclude from weight decay.

  Returns standard patterns for parameters that should not have weight decay applied:
  - "bias": All bias parameters
  - "LayerNorm": All LayerNorm module parameters (weight and bias)
  - "embed": All embedding layer parameters (token embeddings, positional embeddings, etc.)

  Returns:
    Sequence[str]: List of patterns to match against parameter names
  """
  return [".bias", "_bias", ".ln", "_ln", "ln.", "embed"]


def categorize_parameters_by_weight_decay(
    model: nn.Module,
    exclude_patterns: Optional[Sequence[str]] = None
) -> tuple[list, list]:
  """Categorize model parameters into decay and no-decay groups.

  Separates parameters based on exclusion patterns and explicit tagging for proper
  weight decay handling. Parameters can be excluded in two ways:

  1. Pattern matching: Parameters whose names contain any exclusion pattern
  2. Explicit tagging: Parameters with `skip_weight_decay=True` attribute set

  This is useful for non-standard layers that should not have weight decay applied.

  Args:
    model: The model to categorize parameters from
    exclude_patterns: Sequence of strings to match against parameter names.
                     If None, uses default exclusion patterns (bias, LayerNorm, embed)

  Returns:
    tuple: (decay_params, no_decay_params) - Lists of parameters

  Example:
    To mark a specific parameter to skip weight decay in model code:
      param = nn.Parameter(torch.randn(10, 10))
      param.skip_weight_decay = True
      self.register_parameter('custom_param', param)
  """
  if exclude_patterns is None:
    exclude_patterns = get_weight_decay_exclusion_patterns()

  decay_params = []
  no_decay_params = []

  for name, param in model.named_parameters():
    # Check if explicitly tagged to skip weight decay
    skip_decay = hasattr(param, 'skip_weight_decay') and param.skip_weight_decay

    # Check if this parameter matches any exclusion pattern
    pattern_match = any(pattern in name for pattern in exclude_patterns)

    if skip_decay or pattern_match:
      no_decay_params.append(param)
    else:
      decay_params.append(param)

  return decay_params, no_decay_params


def get_default_weight_decay(optimizer_type: str) -> float:
  """Get the default weight decay for each optimizer type.

  Args:
    optimizer_type: Type of optimizer ('adagrad', 'adam', 'adamw', 'sgd', 'rmsprop')

  Returns:
    Default weight decay value for the optimizer type

  Note:
    - AdamW defaults to 1e-2 (0.01) for weight decay
    - All others default to 0.0 (no weight decay)
  """
  optimizer_type = optimizer_type.lower()
  if optimizer_type == 'adamw':
    return 1e-2  # 0.01
  else:  # adagrad, adam, sgd, rmsprop
    return 0.0


def create_optimizer(optimizer_type: str, param_groups: List, lr: float, weight_decay: float = 0.0) -> Optimizer:
  """Create an optimizer instance of the specified type.

  Args:
    optimizer_type: Type of optimizer ('adagrad', 'adam', 'adamw', 'sgd', 'rmsprop')
    param_groups: List of parameter groups for the optimizer
    lr: Learning rate
    weight_decay: Default weight decay (can be overridden per parameter group)

  Returns:
    Optimizer instance

  Raises:
    ValueError: If optimizer_type is not supported
  """
  optimizer_type = optimizer_type.lower()

  if optimizer_type == 'adagrad':
    return torch.optim.Adagrad(param_groups, lr=lr, weight_decay=weight_decay)
  elif optimizer_type == 'adam':
    return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
  elif optimizer_type == 'adamw':
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
  elif optimizer_type == 'sgd':
    return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
  elif optimizer_type == 'rmsprop':
    return torch.optim.RMSprop(param_groups, lr=lr, weight_decay=weight_decay)
  else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                     f"Supported types: adagrad, adam, adamw, sgd, rmsprop")


def detect_freeze(module: nn.Module):
  """Detect and apply gradient freezing based on module/parameter attributes.

  This function looks for 'freeze' attributes on modules and parameters.
  If freeze is True, gradients are disabled for those components.

  Args:
    module: The module to check for freeze attributes
  """
  for m in module.modules():
    if hasattr(m, 'freeze') and m.freeze is not None:
      freeze = m.freeze
      m.requires_grad_(not freeze)
  for p in module.parameters():
    if hasattr(p, 'freeze') and p.freeze is not None:
      freeze = p.freeze
      p.requires_grad_(not freeze)


class NopWith:
  """No-op context manager for conditional context usage.
  
  Used as a placeholder when AMP (automatic mixed precision) is disabled,
  allowing the same code structure regardless of AMP status.
  """
  
  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass


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
    self._model: Optional[LMBase] = None
    self._raw_model: Optional[LMBase] = None
    self._optimizers: Optional[List[Optimizer]] = None
    self.model_stats: Optional[modelstats.ModelStats] = None
    self.step_stats = dict()
    self.current_step = None
    self._model_args = None
    self._optimizer_args = None
    self._stats_dir = stats_dir
    self._lr_scheduler: Optional[LRScheduler] = None
    self.run_name = ""
    self.device: Optional[str] = None
    self.device_type: Optional[str] = None
    self.max_len: Optional[int] = None

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
                 device,
                 locations: Optional[Union[Sequence[str], str]] = None,
                 for_train=True,
                 load_optimizer=True,
                 strict=False,
                 run_name="",
                 default_freeze=False,
                 optimizer_warmup_fraction: Optional[float] = None,
                 optimizer_warmup_steps: Optional[int] = None,
                 disable_optimizer_warmup=False,
                 compile_model=False,
                 compile_mode=None,
                 compile_backend='inductor',
                 amp=False,
                 no_grad_scale=False,
                 reset_history=False,
                 first_step=None,
                 grad_clip=None,
                 check_grads=False,
                 include_prompts=True,
                 **parameters):
    """Initialize the runner with a model and training configuration.
    
    Args:
      device: Device to run on (e.g., 'cuda', 'cpu')
      locations: Optional checkpoint file path(s) to load from
      for_train: Whether to initialize for training (vs inference only)
      load_optimizer: Whether to load optimizer state from checkpoint
      strict: Whether to enforce strict checkpoint loading
      run_name: Name suffix for this run
      default_freeze: Whether to freeze all parameters by default
      optimizer_warmup_fraction: Initial LR as fraction of target
      optimizer_warmup_steps: Number of warmup steps
      disable_optimizer_warmup: Disable LR warmup even if loading checkpoint
      compile_model: Whether to compile model with torch.compile
      compile_mode: Compilation mode (e.g., 'default', 'reduce-overhead')
      compile_backend: Compilation backend (e.g., 'inductor')
      amp: Enable automatic mixed precision
      no_grad_scale: Disable gradient scaling with AMP
      reset_history: Clear training statistics when loading checkpoint
      first_step: Name of first training step
      grad_clip: Gradient clipping value
      check_grads: Print parameters without gradients
      include_prompts: Include prompt tokens in loss calculation
      **parameters: Additional model-specific parameters
    """
    self.include_prompts = include_prompts
    self.check_grads = check_grads
    self.grad_clip = grad_clip
    self.for_train = for_train
    self.device = device
    if 'cuda' in self.device:
      self.device_type = "cuda"
    elif 'mps' in self.device:
      self.device_type = "mps"
    elif 'cpu' in self.device:
      self.device_type = "cpu"
    else:
      self.device_type = device
    # if torch.cuda.is_available():
    #  torch.set_float32_matmul_precision('high')
    torch.set_float32_matmul_precision('high')
    for p in ('lr', 'optimizer_warmup_start', 'optimizer_warmup_steps', 'max_len'):
      if p in parameters and parameters[p] is None:
        del parameters[p]

    self.step_stats = dict()
    self.current_step = first_step

    if len(run_name) > 0:
      self.run_name = f"_{run_name}"
    else:
      self.run_name = ""
    if locations is None:
      locations = []
    if isinstance(locations, str):
      locations = [locations]
    if locations is not None:
      locations = [os.path.expanduser(location) for location in locations]
      locations = [location for location in locations if os.path.exists(location)]

    if len(locations) > 0:
      location = locations[0]
      weight_data = torch.load(location, map_location=device, weights_only=False)
      self._model, self._model_args, missing, unexpected = self._construct_model(device,
                                                                                 model_weights=weight_data.get('model',
                                                                                                               None),
                                                                                 model_args=weight_data.get(
                                                                                   'model_args',
                                                                                   None),
                                                                                 strict=strict,
                                                                                 **parameters)
      if reset_history:
        self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}",
                                                 basedir=self._stats_dir)

      else:
        self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}",
                                                 **weight_data.get('stats', {}),
                                                 basedir=self._stats_dir)
        self.current_step = weight_data.get('current_step', first_step)
        for step_name, data in weight_data.get('step_stats', dict()).items():
          self.step_stats[step_name] = modelstats.ModelStats(
            model_name=f"{self._model.name}{self.run_name}_step_{step_name}",
            **data,
            basedir=self._stats_dir)
        if len(self.step_stats) == 0 and 'stats' in weight_data and first_step != None:
          # looks like we didn't find any step info but there are model stats. We are probably loading an old model.
          # just load the full model stats as the first step.
          self.step_stats[first_step] = modelstats.ModelStats(
            model_name=f"{self._model.name}{self.run_name}_step_{first_step}",
            **weight_data.get('stats', {}),
            basedir=self._stats_dir)

      if for_train:
        if default_freeze:
          self._model.requires_grad_(False)
        detect_freeze(self._model)
        # Only load the other stuff if they are going to train
        self._optimizers, self._optimizer_args, self._lr_scheduler = self.construct_optimizer(device,
                                                                                              self._model,
                                                                                              missing=missing,
                                                                                              unexpected=unexpected,
                                                                                              load_optimizer=load_optimizer,
                                                                                              optimizer_weights=weight_data.get(
                                                                                                'optimizer',
                                                                                                None),
                                                                                              optimizer_args=weight_data.get(
                                                                                                'optimizer_args', None),
                                                                                              disable_optimizer_warmup=disable_optimizer_warmup,
                                                                                              optimizer_warmup_steps=optimizer_warmup_steps,
                                                                                              optimizer_warmup_fraction=optimizer_warmup_fraction,
                                                                                              **parameters)
        if not isinstance(self._optimizers, list):
          self._optimizers = [self._optimizers]

    else:
      self._model, self._model_args = self._construct_model(device, **parameters)
      self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}", basedir=self._stats_dir)
      if for_train:
        if default_freeze:
          self._model.requires_grad_(False)
        detect_freeze(self._model)
        self._optimizers, self._optimizer_args, self._lr_scheduler = self.construct_optimizer(device,
                                                                                              self._model,
                                                                                              **parameters)
        if not isinstance(self._optimizers, list):
          self._optimizers = [self._optimizers]

    self._raw_model = self._model
    if compile_model:
      # Ensure sensible defaults for torch.compile
      # For CUDA, 'inductor' is the recommended backend
      if compile_backend is None:
        compile_backend = 'inductor' if 'cuda' in device else 'eager'
      # For training loops with repeated calls, 'reduce-overhead' mode is better than 'default'
      # Default mode has higher compilation overhead but better kernel fusion
      if compile_mode is None:
        compile_mode = 'reduce-overhead'
      # Compile the model with explicit parameters
      # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
      print(f"Compiling model using {compile_backend}:{compile_mode}")
      self._model = torch.compile(
          self._model,
          backend=compile_backend,
          mode=compile_mode,
          fullgraph=False,  # Allow graph breaks to prevent recompilation issues
          disable=False,    # Enable compilation
      )

    self.scaler = None
    self.amp = NopWith
    if amp:
      if "cuda" in device and not no_grad_scale:
        self.scaler = torch.amp.GradScaler('cuda')
      self.amp = torch.amp.autocast
    self._model.train(for_train)
    self.max_len = self._model.max_len

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
    """Save model checkpoint to disk.
    
    Args:
      location: Path to save the checkpoint
      prod_save: If True, only save model weights (no optimizer/stats)
    """
    assert self.is_initialized(), "Runner not initialized"
    assert self.is_trainable() or prod_save, "Runner not trainable"
    if prod_save:

      checkpoint = {'model': self._raw_model.state_dict(),
                    'model_args': self.get_model_args()}
    else:
      if len(self._optimizers) > 1:
        optimizer_save = [optimizer.state_dict() for optimizer in self._optimizers]
      else:
        optimizer_save = self._optimizers[0].state_dict()

      checkpoint = {'model': self._raw_model.state_dict(),
                    'model_args': self.get_model_args(),
                    'optimizer_args': self.get_optimizer_args(),
                    'optimizer': optimizer_save,
                    'current_step': self.current_step,
                    'stats': self.model_stats.dump_dict(),
                    'step_stats': {stat_name: stat.dump_dict() for stat_name, stat in self.step_stats.items()}}
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
    if self._lr_scheduler:
      self._lr_scheduler.step()
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

  def construct_optimizer(self,
                          device,
                          model: LMBase,
                          missing=None,
                          unexpected=None,
                          load_optimizer=True,
                          optimizer_weights: dict = None,
                          optimizer_args=None,
                          optimizer_warmup_fraction: Optional[float] = None,
                          optimizer_warmup_steps: Optional[int] = None,
                          disable_optimizer_warmup=False,
                          **parameters) -> (Optimizer, Any, Optional[LRScheduler]):
    """Construct one or more optimizers to manage the model.
    
    The first optimizer is the 'primary' and will have scaling and lr scheduling
    applied if available. Secondary optimizers are needed if training on different
    types of devices (CPU + GPU) with AMP. Multiple optimizers are rarely needed.
    
    To assign parameters to a secondary optimizer, tag them with
    'param.secondary_optimizer = True' when they are constructed.
    
    Args:
      device: Device the model is on
      model: The model to optimize
      missing: Missing keys from checkpoint loading
      unexpected: Unexpected keys from checkpoint loading
      load_optimizer: Whether to load optimizer state
      optimizer_weights: Saved optimizer state
      optimizer_args: Saved optimizer arguments
      optimizer_warmup_fraction: Initial LR fraction for warmup
      optimizer_warmup_steps: Number of warmup steps
      disable_optimizer_warmup: Disable warmup scheduler
      **parameters: Additional parameters
      
    Returns:
      tuple: (optimizer(s), optimizer_args, lr_scheduler)
    """

    if optimizer_args is None:
      optimizer_args = dict()
    lr = parameters.get('lr', optimizer_args.get('lr', DEFAULT_LR))

    # Get optimizer type early so we can determine default weight decay
    optimizer_type = parameters.get('optimizer', optimizer_args.get('optimizer', 'adamw')).lower()

    # Get weight decay, handling None by mapping to optimizer-specific default
    weight_decay = parameters.get('weight_decay', optimizer_args.get('weight_decay', None))
    if weight_decay is None:
      weight_decay = get_default_weight_decay(optimizer_type)

    primary_weights = []
    secondary_weights = []
    primary_no_decay_weights = []
    secondary_no_decay_weights = []

    # Detect primary and secondary optimizer targets.
    # If weight_decay > 0, also separate parameters that should not have decay applied
    if weight_decay > 0:
      decay_params, no_decay_params = categorize_parameters_by_weight_decay(model)
      for parameter in decay_params:
        if hasattr(parameter, "secondary_optimizer") and parameter.secondary_optimizer:
          secondary_weights.append(parameter)
        else:
          primary_weights.append(parameter)
      for parameter in no_decay_params:
        if hasattr(parameter, "secondary_optimizer") and parameter.secondary_optimizer:
          secondary_no_decay_weights.append(parameter)
        else:
          primary_no_decay_weights.append(parameter)
    else:
      for parameter in model.parameters():
        if hasattr(parameter, "secondary_optimizer") and parameter.secondary_optimizer:
          secondary_weights.append(parameter)
        else:
          primary_weights.append(parameter)

    # Create parameter groups with appropriate weight decay settings
    # Only use separate groups if weight_decay > 0 and we have parameters to exclude
    param_groups = []

    if weight_decay > 0 and len(primary_no_decay_weights) > 0:
      # Separate decay and no-decay groups only when weight_decay > 0
      if len(primary_weights) > 0:
        param_groups.append({'params': primary_weights, 'weight_decay': weight_decay})
      param_groups.append({'params': primary_no_decay_weights, 'weight_decay': 0.0})
    else:
      # Single group (either weight_decay==0 or no no-decay params)
      all_primary = primary_weights + primary_no_decay_weights if weight_decay > 0 else primary_weights
      if not all_primary:
        all_primary = [p for p in model.parameters() if not hasattr(p, "secondary_optimizer")]
      param_groups.append({'params': all_primary, 'weight_decay': weight_decay})

    optimizers = [create_optimizer(optimizer_type, param_groups, lr, weight_decay)]

    # Handle secondary optimizer with same decay/no-decay logic
    secondary_param_groups = []

    if weight_decay > 0 and len(secondary_no_decay_weights) > 0:
      # Separate decay and no-decay groups only when weight_decay > 0
      if len(secondary_weights) > 0:
        secondary_param_groups.append({'params': secondary_weights, 'weight_decay': weight_decay})
      secondary_param_groups.append({'params': secondary_no_decay_weights, 'weight_decay': 0.0})
    elif len(secondary_weights) > 0 or len(secondary_no_decay_weights) > 0:
      # Single group
      all_secondary = secondary_weights + secondary_no_decay_weights if weight_decay > 0 else secondary_weights
      secondary_param_groups.append({'params': all_secondary, 'weight_decay': weight_decay})

    if secondary_param_groups:
      optimizers.append(create_optimizer(optimizer_type, secondary_param_groups, lr, weight_decay))
    lr_scheduler = None
    if optimizer_weights is not None and not isinstance(optimizer_weights, list):
      optimizer_weights = [optimizer_weights]
    if optimizer_weights is not None and load_optimizer and len(unexpected) == 0 and len(missing) == 0 and len(
            optimizer_weights) == len(optimizers):
      # if optimizer_weights is not None and load_optimizer and len(missing) == 0 and len(unexpected) == 0:
      # only load old optimizers if the model parameters haven't changed.
      optimizer_loaded = True
      for optimizer, weights in zip(optimizers, optimizer_weights):
        # Check if parameter group structure matches to avoid loading issues
        saved_num_groups = len(weights.get('param_groups', []))
        current_num_groups = len(optimizer.param_groups)

        if saved_num_groups != current_num_groups:
          # Parameter group structure has changed (e.g., new weight decay grouping)
          logger.warning(
            f"Parameter group structure changed: saved={saved_num_groups}, current={current_num_groups}. "
            f"Skipping optimizer state loading to avoid mismatch."
          )
          optimizer_loaded = False
          break

        for pg in weights['param_groups']:
          if 'lr' in pg:
            pg['lr'] = lr
          if 'weight_decay' in pg:
            pg['weight_decay'] = weight_decay
        if load_optimizer:
          try:
            optimizer.load_state_dict(weights)
          except (ValueError, RuntimeError) as e:
            optimizer_loaded = False
            logger.error(
              f"Unable to load optimizer state: {e}. Probably a new parameter or structure change. "
              f"(This optimizer doesn't belong to this model)")
        else:
          optimizer_loaded = False
    else:
      optimizer_loaded = False
    create_warmup_scheduler = not disable_optimizer_warmup and optimizer_weights is not None and not optimizer_loaded and len(
      optimizers) == 1
    if create_warmup_scheduler:
      logger.info("Creating learning rate warmup scheduler.")
      lr_scheduler = OptimizerWarmupLRScheduler(optimizers[0],
                                                steps=optimizer_warmup_steps,
                                                initial_fraction=optimizer_warmup_fraction)

    # Log optimizer configuration summary
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
      f"Optimizer setup: {len(optimizers)} instance(s) of {optimizer_type.upper()}, "
      f"total parameters: {total_params:,}, "
      f"lr={lr}, weight_decay={weight_decay}"
    )

    # Store optimizer type in optimizer_args so it persists in checkpoints
    optimizer_args['optimizer'] = optimizer_type

    # Log detailed parameter group information (INFO level for summary, DEBUG for individual params)
    for opt_idx, optimizer in enumerate(optimizers):
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
          param_names = [name for name, p in model.named_parameters() if id(p) in param_ids_in_group]
          for name in param_names:  # Log all parameter names
            logger.debug(f"    - {name}")

    optimizer_args['lr'] = lr
    optimizer_args['weight_decay'] = weight_decay
    if len(optimizers) > 1:
      return optimizers, optimizer_args, lr_scheduler
    return optimizers[0], optimizer_args, lr_scheduler


class BasicModelRunner(LMRunnerBase):
  """Simple model runner implementation for standard models.
  
  This runner provides a straightforward way to wrap a model class
  for training and inference without complex customization.
  
  Args:
    model_class: The model class to instantiate
    max_batch_size: Maximum batch size for gradient accumulation
    overrides: Parameter overrides to apply when loading models
  """
  
  def __init__(self, model_class, max_batch_size=25, overrides: dict = None, **kwargs):
    super().__init__(max_batch_size=max_batch_size, **kwargs)
    self.model_class = model_class
    self.overrides = overrides

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    """Construct model instance from class and parameters.
    
    Args:
      device: Device to place model on
      model_weights: Optional saved weights
      model_args: Optional saved model arguments
      strict: Whether to enforce strict loading
      **parameters: Additional model parameters
      
    Returns:
      tuple: (model instance, model init_kwargs, missing keys, unexpected keys)
            or (model instance, model init_kwargs) if not loading weights
    """

    model_args = model_args if model_args else dict()
    for k, v in parameters.items():
      if k not in model_args or k == "version":
        model_args[k] = v
    if not self.overrides is None:
      for k, v in self.overrides.items():
        # We override with our defaults incase we are starting from a different version model
        model_args[k] = v

    model = self.model_class(**model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
