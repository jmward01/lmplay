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
@expose_runner('rpa1_3',
               description='Trying to line a distil group up on the key/value boundary')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=384,
                                         num_distil_heads=12,
                                         num_distil_head_groups=3),

                          **kwargs)

@expose_runner('rpa1_4',
               description='Trying to make things smaller so they are faster. Seeing how small I can make the key dim.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*3,
                                         num_distil_heads=4,
                                         num_distil_head_groups=2),

                          **kwargs)

@expose_runner('rpa1_5',
               description='Trying to make things smaller so they are faster. Seeing how small I can make the key dim.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*3,
                                         num_distil_heads=4,
                                         num_distil_head_groups=2,
                                         add_model_attn=False),

                          **kwargs)

@expose_runner('rpa1_6',
               description='Trying to make things smaller so they are faster. Seeing how small I can make the key dim.')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*3,
                                         attn_scales=((3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3, 3, 3), (5, 3, 3), (5, 3)),
                                         num_distil_heads=4,
                                         num_distil_head_groups=2,
                                         add_model_attn=False),

                          **kwargs)

@expose_runner('rpa1_7',
               description='Even smaller but with logic at the front')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*3,
                                         attn_scales=((5, 5, 3), (5, 3), (5, 3)),
                                         num_distil_heads=4,
                                         num_distil_head_groups=2,
                                         add_model_attn=False),

                          **kwargs)

@expose_runner('rpa1_8',
               description='Even smaller but with logic at the front')
def runner(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(key_dim=12*3,
                                         attn_scales=(((3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3, 3, 3), (5, 3, 3), (5, 3))),
                                         num_distil_heads=4,
                                         num_distil_head_groups=2,
                                         add_model_attn=False,
                                         mid_mul=3,
                                         front_embed_mul=3),

                          **kwargs)