

from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('foc1_0',
               description='Trying a simple idea to allow the model to learn to focus attn.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('foc1_1',
               description='Trying a less expensive version of this idea.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(scale_type="simple"),
                          **kwargs)
