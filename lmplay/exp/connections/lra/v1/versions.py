from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('lra1_0',
               description='Learned Residual Add. Tries to create a simple learned value for residual adds to allow the network to incorporate data easier.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('lra1_1',
               description='Trying an added bias since the bias on the l and r are in flux.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(add_c=True),
                          **kwargs)