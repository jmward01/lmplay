from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('sac1_0', description="First in the 'sacrificial' line of experiments. These models combine all the sacrificial experiments, experiments that train with extra parameters that are removed for prod")
def sac1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('sac1_0_1', description="Same as sac1_0 but with a 16x front embed")
def sac1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=16),
                          **kwargs)

@expose_runner('sac1_1', description="Same as sac1_0 but with a 16x front embed")
def sac1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(sulinear=True,
                                         front_embed_mul=16),
                          **kwargs)
