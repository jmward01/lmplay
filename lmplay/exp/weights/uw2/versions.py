from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('uw2_0', description="Unifeid Weights using a shared sacrificial network.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)