from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('uw6_0')
def uw6_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(exp_mul=32, #<- big 6.0 diff
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False),
                          **kwargs)


@expose_runner('uw6_1')
def uw6_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False),
                          **kwargs)


@expose_runner('uw6_2')
def uw6_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=None), #<- big 6.2 diff
                          **kwargs)

@expose_runner('uw6_3')
def uw6_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(predict_bias=None, #<- big 6.3 diff
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
                          overrides=dict(exp_mul=32,
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
                          overrides=dict(exp_mul=32,
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=True, #<-big 6.5 diff
                                         ignore_purpose=False),
                          **kwargs)

@expose_runner('uw6_6')
def uw6_6(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(exp_mul=32,
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=True,
                                         ln_fc=False, #<-big diff from last. With all the extra fc logic maybe we don't want the layernorm.
                                         ignore_purpose=False),
                          **kwargs)