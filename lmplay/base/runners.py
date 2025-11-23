from .base_model import LMBase
from .base_runner import LMRunnerBase

class BasicModelRunner(LMRunnerBase):
  """Simple model runner implementation for standard models.

  This runner provides a straightforward way to wrap a model class
  for training and inference without complex customization.

  Args:
    model_class: The model class to instantiate
    mini_batch_size: Maximum batch size for gradient accumulation
    overrides: Parameter overrides to apply when loading models
  """

  def __init__(self, model_class, mini_batch_size=25, overrides: dict = None, **kwargs):
    super().__init__(mini_batch_size=mini_batch_size, **kwargs)
    self.model_class = model_class
    self.overrides = overrides

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, any):
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