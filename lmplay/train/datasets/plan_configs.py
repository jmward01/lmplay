ALL_DATASETS = {'wiki_en': {'ds_loader': 'wiki', 'args': ['en']},
                 'wiki_es': {'ds_loader': 'wiki', 'args': ['es']},
                 'openorca': {'ds_loader': 'openorca'},
                 'orcamathword200': {'ds_loader': 'orcamathword200'},
                 'tinystories': {'ds_loader': 'tinystories'},
                 'smoltalk_all': {'ds_loader':'hf', 'args':['smoltalk_all', 'HuggingFaceTB/smoltalk', 'all'], 'kwargs':{'truth_column':'messages'}}}

DEFAULT_PLAN = {'datasets': ALL_DATASETS,
                'seed': 0,
                'steps': [{'datasets': ['wiki_en', 'wiki_es'],
                           'epochs': 1.0,
                           'step_name': 'wiki_en_es'}, ]}

FULL_V1 = {'datasets': ALL_DATASETS,
           'seed': 0,
           'steps': [{'datasets': ['wiki_en', 'wiki_es'],
                      'epochs': 1.0,
                      'step_name': 'wiki_en_es'},
                     {'datasets': ['tinystories'],
                      'epochs': 1.0,
                      'step_name': 'tinystories'},
                     {'datasets': ['openorca', 'orcamathword200', 'smoltalk_all'],
                      'epochs': 1.0,
                      'step_name': 'instruction_ft'}]}

SMOLTALK_ALL = {'datasets': ALL_DATASETS,
            'seed': 0,
            'steps': [{'datasets': ['smoltalk_all'],
                       'epochs': 1.0,
                       'step_name': 'smoltalk_all'}]}

OPENORCA = {'datasets': ALL_DATASETS,
            'seed': 0,
            'steps': [{'datasets': ['openorca'],
                       'epochs': 1.0,
                       'step_name': 'openorca'}]}


ORCAMATHWORD200 = {'datasets': ALL_DATASETS,
            'seed': 0,
            'steps': [{'datasets': ['orcamathword200'],
                       'epochs': 1.0,
                       'step_name': 'orcamathword200'}]}


TINYSTORIES = {'datasets': ALL_DATASETS,
               'seed': 0,
               'steps': [{'datasets': ['tinystories'],
                          'epochs': 1.0,
                          'step_name': 'tinystories'}]}

DEFAULT_PLANS = {'default': DEFAULT_PLAN,
                 'full_v1': FULL_V1,
                 'openorca': OPENORCA,
                 'tinystories': TINYSTORIES,
                 'orcamathword200':ORCAMATHWORD200,
                 'smoltalk_all': SMOLTALK_ALL}