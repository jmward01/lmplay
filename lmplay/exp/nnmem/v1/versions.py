from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm1_0', "NNMemory! Bolt on memory layers that are vastly more parameter efficient than a full transformer layer and they appear to be more valuable so long as some attn layers are in the mix.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

#
@expose_runner('nnm1_1', description="Tests having individual nnm layer adapters per layer.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(shared_nnm_layer=False),
                          **kwargs)

#
@expose_runner('nnm1_2', description="Tests having individual nnm instances per layer.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(shared_nnm=False),
                          **kwargs)

#
@expose_runner('nnm1_3', description='Tests nnm before the attn.')
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_first=True),
                          **kwargs)

#
@expose_runner('nnm1_4',  description="Adding extra nnm only blocks to see the impact.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(extra_nnm_only_blocks = 1),
                          **kwargs)

#
@expose_runner('nnm1_5', description="Trying a stripped down 6 total layers (3 + 3) to be more comparable to a 6L model. This is fewer parameters/compute than a standard 6L model.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_blocks = 3,
                                         shared_nnm=False),
                          **kwargs)
#
@expose_runner('nnm1_6', "Playing with connections to see if it boosts performance.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_attn_residual=False),
                          **kwargs)

#
@expose_runner('nnm1_7', "Testing how more heads impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_heads=12),
                          **kwargs)

#
@expose_runner('nnm1_8', "Testing how a shorter NNM sequence length impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_size=64),
                          **kwargs)

#
@expose_runner('nnm1_9', "Testing how a larger nnm_emb_mul impacts things.")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nnm_emb_mul=64),
                          **kwargs)

#
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
