from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('uw6_0')
def uw6_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.0",
                                         exp_mul=32, #<- big 6.0 diff
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False),
                          **kwargs)


@expose_runner('uw6_1')
def uw6_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.1",
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False),
                          **kwargs)


@expose_runner('uw6_2')
def uw6_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.2",
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=None), #<- big 6.2 diff
                          **kwargs)

@expose_runner('uw6_3')
def uw6_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.3",
                                         predict_bias=None, #<- big 6.3 diff
                                         exp_mul=32, #<- big 6.3 diff
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False), #<- big 6.3 diff
                          **kwargs)

@expose_runner('uw6_4')
def uw6_4(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.4",
                                         exp_mul=32,
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False,
                                         ignore_purpose=False), #<- big 6.4 diff
                          **kwargs)

@expose_runner('uw6_5')
def uw6_5(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.5",
                                         exp_mul=32,
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=True, #<-big 6.5 diff
                                         ignore_purpose=False),
                          **kwargs)