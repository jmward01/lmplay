from typing import Any

from lmplay.base.base_model import LMBase, LMRunnerBase
from ..unified_weights_v5_1.model import GPT2

class ModelRunner(LMRunnerBase):
  def __init__(self, max_batch_size=25):
    super().__init__(max_batch_size=max_batch_size)

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    model_args = model_args if model_args else dict()
    for k, v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    model = GPT2(for_train=self.for_train,
                 version="5.2",
                 predict_bias=True,
                 predict_mbias=True,
                 predict_mbias2=True,
                 predict_mbias_a=False,
                 predict_mbias2_a=False,
                 ln_attn=False,
                 ln_mlp=False,
                 ln_fc=False, #testing no final FC. If it isn't helping in other places, why would it help here?
                 **model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
