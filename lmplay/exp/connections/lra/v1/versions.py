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
               description='Trying the non-predict full version')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(predict=False),
                          **kwargs)

@expose_runner('lra1_2',
               description='Trying the predict simple version')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(simple=True),
                          **kwargs)

@expose_runner('lra1_3',
               description='Trying the non-predict simple version')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(simple=True,
                                         predict=False),
                          **kwargs)

@expose_runner('lra1_4',
               description='Trying the mlp predict with full')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(predict="mlp"),
                          **kwargs)
