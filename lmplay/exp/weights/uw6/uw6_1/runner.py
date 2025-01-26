from typing import Any
from lmplay.base.base_model import LMBase, LMRunnerBase
from ..model import GPT2

from lmplay.base.runner_list import expose_runner

@expose_runner('uw6_1')
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
    defaults = dict(version="6.1",
                    mmlp=True,
                    share_layers=1,
                    last_activation=False,
                    dl_fc=False)

    model_args = model_args if model_args else dict()
    for k, v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    for k, v in defaults.items():
      #We override with our defaults incase we are starting from a different version model
      model_args[k] = v


    model = GPT2(**model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
