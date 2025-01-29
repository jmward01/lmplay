from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('sac2_0', description="Combines UE 1.0 16x with UW 6.0")
def sac2_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="2.0"),
                          **kwargs)


@expose_runner('sac2_0_med', description="sac2_0 but with 20 heads and embed_dim of 1280")
def sac2_0med(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="2.0.1",
                                         num_heads=20,#<-big diff
                                         embed_dim=1280),#<-big diff
                          **kwargs)


@expose_runner('sac2_1', description="Combines UE 1.0 16x with UW 6.0")
def sac2_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="2.1",
                                         ue_sduw=True),#<-big diff
                          **kwargs)


@expose_runner('sac2_2', description="Combines UE 1.0 16x with UW 6.0")
def sac2_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="2.2",
                                         t_sduw=False),#<-big diff
                          **kwargs)

@expose_runner('sac2_3', description="sac2 but with the final fc using a memory efficient shared deep unified setup and purpose linked shared sacrifical networks.")
def sac2_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="2.3",
                                         ignore_purpose=False, #<-big diff
                                         dl_fc=True),#<-big diff
                          **kwargs)
