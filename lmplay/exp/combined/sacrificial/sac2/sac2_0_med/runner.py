from typing import Any
from lmplay.base.base_model import LMBase, LMRunnerBase
from ..model import GPT2

from lmplay.base.runner_list import expose_runner


@expose_runner('sac2_0_med', description="sac2_0 but with 20 heads and embed_dim of 1280")
class ModelRunner(LMRunnerBase):
  def __init__(self, max_batch_size=25):
    super().__init__(max_batch_size=max_batch_size)

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    # Put changes to defaults here
    defaults = dict(version="2.0.1",
                    num_heads=20,
                    embed_dim=1280)
    model_args = model_args if model_args else dict()
    for k, v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    for k, v in defaults.items():
      # We override with our defaults incase we are starting from a different version model
      model_args[k] = v

    model = GPT2(**model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
