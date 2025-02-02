from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm1_0')
def nnm1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.0"),
                          **kwargs)

@expose_runner('nnm1_1')
def nnm1_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.1",
                                         shared_nnm_layer=False),
                          **kwargs)

@expose_runner('nnm1_2')
def nnm1_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.2",
                                         shared_nnm=False),
                          **kwargs)

@expose_runner('nnm1_3')
def nnm1_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.3",
                                         nnm_first=True),
                          **kwargs)
