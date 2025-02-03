from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm1_0', "NNMemory! Bolt on memory layers that are vastly more parameter efficient than a full transformer layer and they appear to be more valuable so long as some attn layers are in the mix.")
def nnm1_0(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.0",
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_1', description="Tests having individual nnm layer adapters per layer. Looks to be about the same (so far) as having just one shared so probably not worth the extra parameters.")
def nnm1_1(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.1",
                                         shared_nnm_layer=False, #probably not worth it. At least in the early training it is not a big improvement
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_2', description="Tests having individual nnm instances per layer. Looks to be about the same (so far) as having it shared so probably not worth the extra parameters.")
def nnm1_2(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.2",
                                         shared_nnm=False, #probably not worth it. At least in the early training it is not a big improvement
                                         nnm_first=False),
                          **kwargs)

@expose_runner('nnm1_3', description='Tests nnm before the attn. It looks like it is worse with nnm first. So look then think is better than think then look!')
def nnm1_3(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.3",
                                         nnm_first=True), #Looks better with nnm second
                          **kwargs)

@expose_runner('nnm1_4',  description="Adding extra nnm only blocks to see the impact. Yep, makes it better.")
def nnm1_4(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.4",
                                         extra_nnm_only_blocks = 1), #Shows clear improvement with another block. Now ties with 20 layer for a lot less cost.
                          **kwargs)

@expose_runner('nnm1_5', description="Trying a stripped down 6 total layers (2 + 4) with only two standard attn layers to see how well it competes with a standarad 6 layer.")
def nnm1_5(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.5",
                                         extra_nnm_only_blocks = 3,
                                         num_blocks = 2),
                          **kwargs)

#nnm_attn_residual
@expose_runner('nnm1_6', "Playing with connections to see if it boosts performance.")
def nnm1_6(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(version="1.6",
                                         nnm_attn_residual=False),
                          **kwargs)