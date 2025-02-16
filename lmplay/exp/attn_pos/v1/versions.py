from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('apos1_0', "Attn with Position v1_0! My theory is that position matters more than you think. This adds a revers position component to the attn calculation.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mul=False),
                          **kwargs)

@expose_runner('apos1_1', "Trying a mul instead of an add to allow position to have a bigger impact.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mul=True),
                          **kwargs)