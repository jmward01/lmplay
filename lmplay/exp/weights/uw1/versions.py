from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('uw1_0', description="Unifeid Weights using the basic mbias, mbias_bias and bias_bias")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('uw1_1', description="Turning on imbias")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(imbias=True),
                          **kwargs)

@expose_runner('uw1_2', description="Turning on imbias")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(imbias=True,
                                         iambias=True,
                                         ambias=True),
                          **kwargs)