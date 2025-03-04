from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


#the cli name will be passed in as version so no need to pass it to the model. You can force a version overrid if you want it different from the cli.
@expose_runner('rgpt2ish',
               description='The reference model class for recurrent GPT models. It ignores the recurrent state and uses the normal GPT2ish implementation. It is shere to show that training isn'' impacted by the recurrent training')
def gpt2ish(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)