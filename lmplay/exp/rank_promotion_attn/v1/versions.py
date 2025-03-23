from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


#the cli name will be passed in as version so no need to pass it to the model. You can force a version overrid if you want it different from the cli.
@expose_runner('rpa1_0',
               description='Time to get a bit crazy.')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)