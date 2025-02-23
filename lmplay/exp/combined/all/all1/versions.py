from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('all1_0', description="Combines UE 1.0 16x, UW 6.0, LRAdd and vnorm")
def sac2_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('all1_1', description="All1_1 but with a different b_min.")
def sac2_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(b_min=0.2),
                          **kwargs)
