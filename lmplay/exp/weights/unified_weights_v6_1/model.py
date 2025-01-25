from typing import Any
from lmplay.base.base_model import LMBase, LMRunnerBase
from ..unified_weights_v6_0.model import GPT2


class ModelRunner(LMRunnerBase):
  def __init__(self, max_batch_size=25):
    super().__init__(max_batch_size=max_batch_size)

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    defaults = dict(version="6.1",
                    exp_mul=8.0,  # <- this is 6.1
                    predict_bias=True,
                    predict_mbias=True,
                    predict_mbias2=True,
                    predict_mbias_a=None,
                    predict_mbias2_a=None,
                    ln_attn=False,
                    ln_mlp=False,
                    ln_fc=True,
                    dl_fc=True,
                    share_in=True,
                    share_out=True,
                    ulinear=False,
                    share_layers=2,
                    share_mid_mul=8.0)
    model_args = model_args if model_args else dict()  # <- this is 6.1
    for k, v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    for k, v in defaults.items():
      if k not in model_args:
        model_args[k] = v
    model = GPT2(**model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
