from .common import Component
from .common_runner import LMRunnerCommon
from lmplay.base.defaults import DEFAULT_LR
from lmplay.base.optimizer import get_default_weight_decay, categorize_parameters_by_weight_decay, create_optimizer
import logging
from lmplay.base.optimizer import detect_freeze
logger = logging.getLogger(__name__)


class OptimizersComponent(Component):
  def __init__(self, mr: LMRunnerCommon):
    super().__init__()
    self.mr = mr
    self._optimizer_args = dict()


  def archive(self) -> dict:
    """Archive optimizer state.

    Saves the optimizer configuration (state_args) and state. Always returns
    optimizer states as a list for consistency.

    Returns:
      dict with 'state_args' (optimizer config) and 'state' (list of optimizer states)
    """
    optimizer_states = [optimizer.state_dict() for optimizer in self.mr._optimizers]

    return {
            'state_args': self._optimizer_args,
            'state': optimizer_states
    }

  def advertise(self) -> dict:
    """Advertise optimizer construction parameters.

    Returns the current optimizer configuration (type, lr, weight_decay, etc.)
    so it can be exposed in config files.

    Returns:
      dict of optimizer configuration parameters
    """
    advertise = self._optimizer_args.copy()
    if 'default_freeze' not in advertise:
      advertise['default_freeze'] = False
    return advertise

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Construct optimizer from checkpoint or fresh.

    Sets internal state: self.mr._optimizers, self._optimizer_args

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
    if not self.mr._model:
      raise RuntimeError("Model must be constructed before optimizers")
    if not self.mr.for_train:
      #Nothing to construct if we aren't training
      return
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
    decay_params, no_decay_params = categorize_parameters_by_weight_decay(self.mr._model)
    param_groups = []

    if weight_decay > 0 and len(no_decay_params) > 0:
      if len(decay_params) > 0:
        param_groups.append({'params': decay_params, 'weight_decay': weight_decay})
      param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
    else:
      all_params = decay_params + no_decay_params if weight_decay > 0 else decay_params
      if not all_params:
        all_params = list(self.mr._model.parameters())
      param_groups.append({'params': all_params, 'weight_decay': weight_decay})

    default_freeze = construction_args.get('default_freeze', False)
    # Handle parameter freezing for training
    if self.mr.for_train:
      # Apply default freeze if configured
      if default_freeze:
        logger.info("Applying default_freeze: freezing all model parameters")
        self.mr._model.requires_grad_(False)
      detect_freeze(self.mr._model)


    # Create optimizer
    optimizer = create_optimizer(optimizer_type, param_groups, lr, weight_decay)
    self.mr._optimizers = [optimizer]

    # Load optimizer state if available and allowed
    optimizer_weights = state if (load_optimizer and state) else None
    if optimizer_weights:
      if not isinstance(optimizer_weights, list):
        optimizer_weights = [optimizer_weights]

      # Only load if model didn't change (no missing/unexpected keys) and counts match
      if not missing and not unexpected and len(optimizer_weights) == len(self.mr._optimizers):
        for opt_idx, (optimizer, weights) in enumerate(zip(self.mr._optimizers, optimizer_weights)):
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
            # weight decay is always 0 for second group. This should be bias/other params that weight decay hurts.
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
    total_params = sum(p.numel() for p in self.mr._model.parameters())
    logger.info(
            f"Optimizer setup: {len(self.mr._optimizers)} instance(s) of {optimizer_type.upper()}, "
            f"total parameters: {total_params:,}, "
            f"lr={lr}, weight_decay={weight_decay}"
    )

    # Store optimizer type in optimizer_args so it persists in checkpoints
    self._optimizer_args['optimizer'] = optimizer_type
    self._optimizer_args['lr'] = lr
    self._optimizer_args['weight_decay'] = weight_decay

    # Log detailed parameter group information (INFO level for summary, DEBUG for individual params)
    for opt_idx, optimizer in enumerate(self.mr._optimizers):
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
          param_names = [name for name, p in self.mr._model.named_parameters() if id(p) in param_ids_in_group]
          for name in param_names:  # Log all parameter names
            logger.debug(f"    - {name}")