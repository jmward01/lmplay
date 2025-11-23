from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('apos1_0', "Attn with Position v1_0! My theory is that position matters more than you think. This adds a revers position component to the attn calculation.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('apos1_1', "Trying a mul instead of an add to allow position to have a bigger impact.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mul=True),
                          **kwargs)

@expose_runner('apos1_2', "Trying just the last 2 blocks as pos attn. The thought here is initial attn is unbiased but thought is focused at the end.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(final_apos=2),
                          **kwargs)
