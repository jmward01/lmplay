from lmplay.base.base_model import BasicModelRunner
from .model import GPT2, RGPT2
from lmplay.base.runner_list import expose_runner



@expose_runner('rgpt2ish',
               description='The reference model class for recurrent GPT models. It ignores the recurrent state and uses the normal GPT2ish implementation. It is shere to show that training isn'' impacted by the recurrent training')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(RGPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('gpt2ish',
               description='The reference model class all other models are compared to. Loosely based on GPT2 so any mistakes are mine not theirs.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)


@expose_runner('gpt2ish_med', description='24layer 1024 embedding 16 heads.')
def gpt2ish_med(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=16,
                                         num_blocks=24,
                                         embed_dim=1024),
                          **kwargs)


@expose_runner('gpt2ish_large', description='36 layer 1280 embedding 20 heads.')
def gpt2ish_large(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=20,
                                         num_blocks=36,
                                         embed_dim=1280),
                          **kwargs)

@expose_runner('test_runner', description='Tiny! Built to just test the plumbing quickly.')
def test_runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(num_heads=2,
                                         #num_blocks=0,
                                         embed_dim=64),
                          **kwargs)
