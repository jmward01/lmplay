from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm2_0', "This uses NNMemory as a layer. Totally crazy idea but it could be interesting!")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('nnm2_1', "Now with shadow!")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(shadow=True),
                          **kwargs)
