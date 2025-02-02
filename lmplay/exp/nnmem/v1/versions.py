from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm1_0')
def nnm1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.0",
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_1')
def nnm1_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.1",
                                         shared_nnm_layer=False,
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_2')
def nnm1_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.2",
                                         shared_nnm=False,
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_3')
def nnm1_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.3",
                                         nnm_first=True), #<-this version showed this was better. Now the default.
                          **kwargs)

@expose_runner('nnm1_4')
def nnm1_4(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.4",
                                         extra_nnm_only_blocks = 1),
                          **kwargs)
