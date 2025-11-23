from .common import Component
from .common_runner import LMRunnerCommon


class ModelComponent(Component):
  def __init__(self, mr: LMRunnerCommon):
    super().__init__()
    self.mr = mr
    self._model_args = dict()


  def archive(self) -> dict:
    """Archive model state.

    Saves the model's construction arguments (state_args) and weights (state).
    state_args represents the parameters that were used to construct the model,
    while state contains the learned weights and internal tensors.

    Returns:
      dict with 'state_args' (construction args) and 'state' (model weights)
    """
    return {
            'state_args': self._model_args if self._model_args else {},
            'state': self.mr._raw_model.state_dict()
    }

  def advertise(self) -> dict:
    """Advertise model construction parameters.

    Asks the model what construction parameters it supports, so they can
    be exposed in config files. This calls the model's advertise_params() method.

    Returns:
      dict of supported construction parameters
    """
    return self.mr._model.advertise_params()

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Construct model from checkpoint or fresh.

    Sets internal state: self.mr._model, self.mr._raw_model, self._model_args, self.mr.max_len

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
      result = self.mr._construct_model(
              self.mr.device,
              model_weights=state,
              model_args=merged_args,
              strict=False,
              **merged_args
      )
      # Unpack result - with weights we may get 4 values
      if isinstance(result, tuple) and len(result) == 4:
        self.mr._model, self._model_args, missing, unexpected = result
      else:
        self.mr._model, self._model_args = result[0], result[1]
    else:
      # Fresh model without weights
      result = self.mr._construct_model(self.mr.device, **merged_args)
      if isinstance(result, tuple) and len(result) >= 2:
        self.mr._model, self._model_args = result[0], result[1]
      else:
        self.mr._model = result
        self._model_args = merged_args

    self.mr._raw_model = self.mr._model
    self.mr.max_len = self.mr._model.max_len