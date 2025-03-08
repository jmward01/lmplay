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
                          overrides=dict(front_embed_mul=16.0),
                          **kwargs)

@expose_runner('ue1x', description="Unifeid Embeddings with a 1x front embedding multiplier")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=1.0),
                          **kwargs)

@expose_runner('ue1_4', description="1x front end emb but using a ulinear")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=1.0,
                                         linear='u'),
                          **kwargs)
@expose_runner('ue1_5', description="Position embeddings")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(pos_embed=True,
                                         tok_embed=False),
                          **kwargs)

@expose_runner('ue1_6', description="Front mul 4.0. Mid mul 64.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=4.0,
                                         mid_mul=64.0),
                          **kwargs)

@expose_runner('ue1_7', description="Front mul 16.0. Mid mul 4.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=16.0,
                                         mid_mul=4.0),
                          **kwargs)

@expose_runner('ue1_8', description="Front mul 8.0. Mid mul 32.")
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(front_embed_mul=8.0,
                                         mid_mul=32.0),
                          **kwargs)