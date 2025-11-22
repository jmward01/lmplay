"""GPT2ish model runner definitions.

This module defines multiple versions of the GPT2-like baseline model with different
configurations (small, medium, large, etc.). Each version is registered as a CLI command
via the @expose_runner decorator, making it available to the training framework.

The runners wrap the GPT2 model class with specific architecture configurations suitable
for different hardware constraints and experimentation goals.
"""

from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('gpt2ish',
               description='The reference model class all other models are compared to. Loosely based on GPT2 so any mistakes are mine not theirs.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('gpt2ish_med', description='24layer 1024 embedding 16 heads.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=16,
                                         num_blocks=24,
                                         embed_dim=1024),
                          **kwargs)


@expose_runner('gpt2ish_large', description='36 layer 1280 embedding 20 heads.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=20,
                                         num_blocks=36,
                                         embed_dim=1280),
                          **kwargs)

@expose_runner('test_runner', description='Tiny! Built to just test the plumbing quickly.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=2,
                                         #num_blocks=0,
                                         embed_dim=64),
                          **kwargs)

@expose_runner('s_gpt2ish', description='Trying small and deep models')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=6,
                                         num_blocks=18,
                                         embed_dim=96*3),
                          **kwargs)
