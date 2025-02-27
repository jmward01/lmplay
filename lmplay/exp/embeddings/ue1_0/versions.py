from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('ue8x',
               description="Unified Embeddings with an 8x front embedding multiplier. A Unified Embedding puts FF in front of a large 'front' embedding.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('ue16x', description="Unifeid Embeddings with a 16x front embedding multiplier")

def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=16.0,
                                         num_blocks=24,
                                         embed_dim=1024),
                          **kwargs)