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


@expose_runner('rpa1_1',
               description='Trying more scale layers that are smaller.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(attn_scales=(5, 5, 5, 5, 10)),
                          **kwargs)


@expose_runner('rpa1_2',
               description='Adding attn pos attn.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(attn_scales=(5, 5, 5, 5, 10),
                                         add_attn_postion=True),

                          **kwargs)

@expose_runner('rpa1_3',
               description='Pos attn and switching around the attn scales.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(attn_scales=(5, 5, 10, 5, 5),
                                         add_attn_postion=True),

                          **kwargs)
