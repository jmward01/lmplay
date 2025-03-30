ALL_DATASETS = {'wiki_en': {'ds_loader': 'wiki', 'args': ['en']},
                'wiki_es': {'ds_loader': 'wiki', 'args': ['es']},
                'openorca': {'ds_loader': 'openorca'},
                'orcamathword200': {'ds_loader': 'hf',
                                    'args': ['orcamathword200', 'microsoft/orca-math-word-problems-200k'],
                                    'kwargs': {'truth_column': 'answer', 'prompt_column': 'question'}},
                'tinystories': {'ds_loader': 'hf',
                                'args': ['tinystories', 'roneneldan/TinyStories'],
                                'kwargs': {'truth_column': 'text'}},
                'smoltalk_all': {'ds_loader': 'hf',
                                 'args': ['smoltalk_all', 'HuggingFaceTB/smoltalk', 'all'],
                                 'kwargs': {'truth_column': 'messages'}},
                'cosmopedia_v2': {'ds_loader': 'hf',
                                  'args': ['cosmopedia_v2', 'HuggingFaceTB/smollm-corpus', 'cosmopedia-v2'],
                                  'kwargs': {'text_column': 'text'}}}

# Too big
# 'fineweb_edu': {'ds_loader': 'hf',
#                 'args': ['fineweb_edu', 'HuggingFaceTB/smollm-corpus', 'fineweb-edu-dedup'],
#                 'kwargs': {'text_column': 'text'}}}


WIKI_EN_ES_STEP = {'datasets': ['wiki_en', 'wiki_es'],
                   'epochs': 1.0,
                   'step_name': 'wiki_en_es'}

LARGE_WIKI_EN_ES_STEP = {'datasets': ['wiki_en', 'wiki_es'],
                         'epochs': 1.0,
                         'step_name': 'l_wiki_en_es'}

PRETRAIN_STEP = {'datasets': ['tinystories'],
                 'epochs': 1.0,
                 'step_name': 'pretrain'}
BIG_PRETRAIN_STEP = {'datasets': ['tinystories', 'cosmopedia_v2'],
                     'epochs': 1.0,
                     'step_name': 'bigpretrain'}

HUGE_PRETRAIN_STEP = {'datasets': ['tinystories', 'cosmopedia_v2'],
                      'epochs': 1.0,
                      'step_name': 'hugepretrain',
                      'interleve_stopping_strategy': "all_exhausted"}

INSTRUCTION_FT_STEP = {'datasets': ['openorca', 'orcamathword200', 'smoltalk_all'],
                       'epochs': 1.0,
                       'step_name': 'l_instruction_ft'}

LARGE_INSTRUCTION_FT_STEP = {'datasets': ['openorca', 'orcamathword200', 'smoltalk_all'],
                             'epochs': 1.0,
                             'step_name': 'l_instruction_ft',
                             'interleve_stopping_strategy': "all_exhausted"}

ALL_STEP = {'datasets': list(ALL_DATASETS),
            'epochs': 1.0,
            'step_name': 'all',
            'interleve_stopping_strategy': "all_exhausted"}

DEFAULT_PLAN = {'datasets': ALL_DATASETS,
                'seed': 0,
                'steps': [WIKI_EN_ES_STEP, ]}

WIKI_EN_ES = {'datasets': ALL_DATASETS,
              'seed': 0,
              'steps': [WIKI_EN_ES_STEP, ]}

LARGE_WIKI_EN_ES = {'datasets': ALL_DATASETS,
              'seed': 0,
              'steps': [LARGE_WIKI_EN_ES_STEP, ]}


PRETRAIN = {'datasets': ALL_DATASETS,
            'seed': 0,
            'steps': [PRETRAIN_STEP, ]}

BIG_PRETRAIN = {'datasets': ALL_DATASETS,
                'seed': 0,
                'steps': [BIG_PRETRAIN_STEP, ]}

HUGE_PRETRAIN = {'datasets': ALL_DATASETS,
                'seed': 0,
                'steps': [HUGE_PRETRAIN_STEP, ]}

INSTRUCTION_FT = {'datasets': ALL_DATASETS,
                  'seed': 0,
                  'steps': [INSTRUCTION_FT_STEP, ]}

LARGE_INSTRUCTION_FT = {'datasets': ALL_DATASETS,
                  'seed': 0,
                  'steps': [LARGE_INSTRUCTION_FT_STEP, ]}

FULL_V1 = {'datasets': ALL_DATASETS,
        'seed': 0,
        'steps': [WIKI_EN_ES_STEP,
                  BIG_PRETRAIN_STEP,
                  INSTRUCTION_FT_STEP]}


FULL = {'datasets': ALL_DATASETS,
        'seed': 0,
        'steps': [LARGE_WIKI_EN_ES_STEP,
                  HUGE_PRETRAIN_STEP,
                  LARGE_INSTRUCTION_FT_STEP]}

SMOLTALK_ALL = {'datasets': ALL_DATASETS,
                'seed': 0,
                'steps': [{'datasets': ['smoltalk_all'],
                           'epochs': 1.0,
                           'step_name': 'smoltalk_all'}]}

COSMOPEDIA_V2 = {'datasets': ALL_DATASETS,
                 'seed': 0,
                 'steps': [{'datasets': ['cosmopedia_v2'],
                            'epochs': 1.0,
                            'step_name': 'cosmopedia_v2'}]}

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

ALL = {'datasets': ALL_DATASETS,
       'seed': 0,
       'steps': [ALL_STEP]}

DEFAULT_PLANS = {'default': DEFAULT_PLAN,
                 'cosmopedia_v2': COSMOPEDIA_V2,
                 'wiki_en_es': WIKI_EN_ES,
                 'l_wiki_en_es': LARGE_WIKI_EN_ES,
                 'full_v1': FULL_V1,
                 'full': FULL,
                 'bigpretrain': BIG_PRETRAIN,
                 'hugepretrain': HUGE_PRETRAIN,
                 'openorca': OPENORCA,
                 'tinystories': TINYSTORIES,
                 'orcamathword200': ORCAMATHWORD200,
                 'smoltalk_all': SMOLTALK_ALL,
                 'instruction_ft': INSTRUCTION_FT,
                 'l_instruction_ft': LARGE_INSTRUCTION_FT,
                 'pretrain': PRETRAIN,
                 'all': ALL, }
# 'fineweb_edu':FINWEB_EDU}
