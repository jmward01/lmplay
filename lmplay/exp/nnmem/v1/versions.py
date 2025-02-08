from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm1_0', "NNMemory! Bolt on memory layers that are vastly more parameter efficient than a full transformer layer and they appear to be more valuable so long as some attn layers are in the mix.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_first=False),
                          **kwargs)

# Minor improvement over baseline and not nearly as big of an improvement as 1.2.
@expose_runner('nnm1_1', description="Tests having individual nnm layer adapters per layer.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(shared_nnm_layer=False, #probably not worth it. At least in the early training it is not a big improvement
                                         nnm_first=False),
                          **kwargs)

#Noticable improvement. (step 1)
@expose_runner('nnm1_2', description="Tests having individual nnm instances per layer.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(shared_nnm=False,
                                         nnm_first=False),
                          **kwargs)
# It looks like it is slightly worse with nnm first (partial step 1). So look then think is better than think then look!
@expose_runner('nnm1_3', description='Tests nnm before the attn.')
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_first=True),
                          **kwargs)

#1.4 is a major improvement. Increasingly better than 20l in step 1 but step 2 is a little worse than 20l, but maintains.
@expose_runner('nnm1_4',  description="Adding extra nnm only blocks to see the impact.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(extra_nnm_only_blocks = 1), #Shows clear improvement with another block. Now ties with 20 layer for a lot less cost.
                          **kwargs)

#Not nearly as good as a standard 6l baseline. (step 1)
@expose_runner('nnm1_5', description="Trying a stripped down 6 total layers (2 + 4) with only two standard attn layers to see how well it competes with a standarad 6 layer.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(extra_nnm_only_blocks = 3,
                                         num_blocks = 2),
                          **kwargs)
#Much worse. (partial step 1)
@expose_runner('nnm1_6', "Playing with connections to see if it boosts performance.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_attn_residual=False),
                          **kwargs)

#Likely small, but noticable, improvement (step 1)
@expose_runner('nnm1_7', "Testing how more heads impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_heads=12),
                          **kwargs)

#Looks to be slowly falling behind. (step 1)
@expose_runner('nnm1_8', "Testing how a shorter NNM sequence length impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_size=64),
                          **kwargs)

#Possibly slightly worse (early step 1)
# I think this is because of two reasons:
# - No token drag correction since all values are used every example
# - The UE restricts down to emb size then the adaptors hit on top of that so too many layers in the way.
@expose_runner('nnm1_9', "Testing how a larger nnm_emb_mul impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_emb_mul=64),
                          **kwargs)

#No noticable difference so success!
# Fewer training parameters and same performance. This should probably be the default.
@expose_runner('nnm1_10', "Testing without UE.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_emb_mul=0),
                          **kwargs)


@expose_runner('nnm1_11', "Testing without UE and with a bigger embedding to share between the k and v.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_emb_mul=0,
                                         nnm_emb_mul2=8),
                          **kwargs)
