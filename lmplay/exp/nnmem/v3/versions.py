from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm3_0', "Trying a much cheaper idea and applying it to weights.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('nnm3_1', "Seing what happens with more cells (20 instead of 10).")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(cells=20),
                          **kwargs)

@expose_runner('nnm3_2', "Trying nnmlinear on just the ff in the block.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mha_linear=False),
                          **kwargs)

@expose_runner('nnm3_3', "Trying mha linear only.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(ff_linear=False),
                          **kwargs)
