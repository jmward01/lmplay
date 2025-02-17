from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm3_0', "Trying a much cheaper idea and applying it to weights.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('nnm3_1', "Seing what happens with more cells (64 instead of 10).")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(cells=64),
                          **kwargs)
