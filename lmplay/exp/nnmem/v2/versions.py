from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm2_0', "This uses NNMemory as a layer. Totally crazy idea but it could be interesting!")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_first=False),
                          **kwargs)
