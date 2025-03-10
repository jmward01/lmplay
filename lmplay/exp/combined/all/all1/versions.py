from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('all1_0', description="Combines UE 1.0 16x, UW 6.0, LRAdd and vnorm")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('all1_1', description="Trying out mlp and different floor ceil.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False,
                                         lradd_floor=0.7,
                                         lradd_ceil=1.4),
                          **kwargs)


@expose_runner('all1_2', description="Trying a higher lradd_floor")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False,
                                         lradd_floor=0.6,
                                         lradd_ceil=1.4),
                          **kwargs)


@expose_runner('all1_3', description="ULinear bumpped up")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False,
                                         lradd_floor=0.6,
                                         lradd_ceil=1.4,
                                         imbias=True,
                                         iambias=True,
                                         ambias=True,
                                         mulinear=True),
                          **kwargs)

@expose_runner('all1_4', description="all1_3 but with the front emb mul at 8 and the mid mul at 8.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(lradd_predict="mlp",
                                         lradd_simple=False,
                                         lradd_floor=0.6,
                                         lradd_ceil=1.4,
                                         imbias=True,
                                         iambias=True,
                                         ambias=True,
                                         mulinear=True,
                                         front_embed_mul=8),
                          **kwargs)
                                         #mid_mul=32,
                                         #ue_sduw=False,
                                         #norm_v=False),
                          #**kwargs)