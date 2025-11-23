

from lmplay.base.runners import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


#not really sure what I was thinking here. Meh! Try things and don't think about them too hard unless they work!
@expose_runner('foc1_0',
               description='Trying a simple idea to allow the model to learn to focus attn.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

#Much simpler thing here. If scaling is really important than allowing it to float to a good value is possibly a good thing.
#A minor improvement at the very beginning of training but it doesn't look like it lasts.
# Didn't run it long since it isn't worth the cost of not being able to use the optimized pytorch version
@expose_runner('foc1_1',
               description='Trying a less expensive version of this idea.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(scale_type="simple"),
                          **kwargs)
