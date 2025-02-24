from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('all1_0', description="Combines UE 1.0 16x, UW 6.0, LRAdd and vnorm")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('all1_1', description="Trying out mlp with full.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False),
                          **kwargs)

@expose_runner('all1_2', description="Trying a higher lradd_floor")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False,
                                         lradd_floor=0.6,
                                         lradd_ceil=1.4),
                          **kwargs)
