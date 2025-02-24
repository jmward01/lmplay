from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('sac2_0', description="Combines UE 1.0 16x with UW 6.0")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('sac2_0_med', description="sac2_0 but with 20 heads and embed_dim of 1280")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=20,#<-big diff
                                         embed_dim=1280),#<-big diff
                          **kwargs)


@expose_runner('sac2_1', description="Combines UE 1.0 16x with UW 6.0")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(ue_sduw=True),#<-big diff
                          **kwargs)


@expose_runner('sac2_2', description="Combines UE 1.0 16x with UW 6.0")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(t_sduw=False),#<-big diff
                          **kwargs)

@expose_runner('sac2_3', description="sac2 but with the UE and final fc using a memory efficient shared deep unified setup and purpose linked shared sacrifical networks.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(max_predict=True,
                                         ue_sduw=True, #<-big diff
                                         ignore_purpose=False, #<-big diff
                                         dl_fc=True),#<-big diff
                          **kwargs)

@expose_runner('sac2_4', description="sac2 2.3 but dl_fc back to false! It looks like it initially helps but then eventually hurts. Best guess is this is because it just learns to drop loss and doesn't actually add ability to learn knowledge.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(max_predict=True,
                                         ue_sduw=True,
                                         ignore_purpose=False),
                          **kwargs)
