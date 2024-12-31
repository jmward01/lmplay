from typing import Any

from lmplay.base.base_model import LMBase, LMRunnerBase
from ..unified_embeddings_v1_0.model import GPT2

class ModelRunner(LMRunnerBase):
  def __init__(self, max_batch_size=25):
    super().__init__(max_batch_size=max_batch_size)

  #This is the same as 1.0 except the embed is 16 instead of 8 and because it is so large it is kept on the CPU by default to save memory at the cost of training speed.
  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    model_args = model_args if model_args else dict(front_embed_mul=16.0)
    for k,v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    model = GPT2(for_train=self.for_train, keep_embed_on_cpu=True, version="1.1", **model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
