from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm4_0', "Trying 1.0 (sort of) with the cheaper NNMem")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('nnm4_1', "nn_ln set to False to be closer to the old nnm1.0")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nn_ln=False),
                          **kwargs)
