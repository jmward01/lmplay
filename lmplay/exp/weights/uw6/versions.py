from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('uw6_0')
def uw6_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.0",
                                         exp_mul=32,
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
                                         dl_fc=None),
                          **kwargs)

@expose_runner('uw6_3')
def uw6_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="6.3",
                                         predict_bias=None,
                                         exp_mul=32,
                                         mmlp=True,
                                         share_layers=1,
                                         last_activation=False,
                                         dl_fc=False),
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
                                         ignore_purpose=False),
                          **kwargs)