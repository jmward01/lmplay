from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


# the cli name will be passed in as version so no need to pass it to the model. You can force a version overrid if you want it different from the cli.
@expose_runner('rpa1_0',
               description='Time to get a bit crazy.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

#Generally works with a little more performance/less mem but a bit less accuracy. Bascially as expected.
@expose_runner('rpa1_1',
               description='Trying a smaller key size')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*20),

                          **kwargs)


@expose_runner('rpa1_2',
               description='Trying a smaller key size but more heads!')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*20,
                                         num_distil_heads=12,
                                         num_distil_head_groups=3),

                          **kwargs)
#num_distil_head_groups