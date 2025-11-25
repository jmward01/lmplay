from typing import Optional, Sequence, List

import torch
from torch import nn
from torch.optim import Optimizer
from lmplay.base.defaults import DEFAULT_ADAMW_WEIGHT_DECAY, DEFAULT_WEIGHT_DECAY
import logging
logger = logging.getLogger(__name__)

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
    exclude_patterns = model.weight_decay_exclusion_patterns
  logger.info(f"Excluding these patterns from weight decay: {', '.join(exclude_patterns)}")
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
    return DEFAULT_ADAMW_WEIGHT_DECAY
  else:  # adagrad, adam, sgd, rmsprop
    return DEFAULT_WEIGHT_DECAY


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
