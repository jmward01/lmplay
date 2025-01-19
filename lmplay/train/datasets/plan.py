from .lmpdatasets import get_dataset
from datasets import interleave_datasets, Dataset

_ALL_DATASETS = {'wiki_en': {'name': 'wiki', 'args': ['en']},
                              'wiki_es': {'name': 'wiki', 'args': ['es']},
                              'openorca': {'name': 'openorca'},
                 'tinystories': {'name': 'tinystories'}}

DEFAULT_PLAN = {'datasets': _ALL_DATASETS,
                'seed': 0,
                'steps': [{'datasets': ['wiki_en', 'wiki_es'],
                           'epochs': 1.0,
                           'step_name': 'wiki_en_es'}, ]}

FULL_V1 = {'datasets': _ALL_DATASETS,
           'seed': 0,
           'steps': [{'datasets': ['wiki_en', 'wiki_es'],
                      'epochs': 1.0,
                      'step_name': 'wiki_en_es'},
                     {'datasets': ['tinystories'],
                      'epochs': 1.0,
                      'step_name': 'tinystories'},
                     {'datasets': ['openorca'],
                      'epochs': 1.0,
                      'step_name': 'openorca'}]}

OPENORCA = {'datasets': _ALL_DATASETS,
           'seed': 0,
           'steps': [{'datasets': ['openorca'],
                      'epochs': 1.0,
                      'step_name': 'openorca'}]}

TINYSTORIES = {'datasets': _ALL_DATASETS,
           'seed': 0,
           'steps': [{'datasets': ['tinystories'],
                      'epochs': 1.0,
                      'step_name': 'tinystories'}]}



DEFAULT_PLANS = {'default': DEFAULT_PLAN,
                 'full_v1': FULL_V1,
                 'openorca':OPENORCA,
                 'tinystories':TINYSTORIES}


def get_step_names(step_def: dict):
  steps = step_def['steps']
  return tuple(step.get('step_name', str(i)) for i, step in enumerate(steps))


def get_first_step_name(step_def: dict):
  return get_step_names(step_def)[0]


# A plan is simple. It defines datasets and training steps.
# Each step consists of datasets that will be randomized then interleaved and run for a given number/fraction of an epoch.
def steps(step_def: dict, save_dataset=False, current_step: str = None) -> (str, Dataset, Dataset):
  seed = step_def.get('seed', 0)
  found_current = current_step is None
  for i, step in enumerate(step_def['steps']):
    step_epochs = step.get('epochs', 1.0)
    step_name = step.get('step_name', str(i))
    # Fast forward past steps we have already done.
    if current_step == step_name:
      found_current = True

    if found_current:
      print(f"Loading {step_name} for {step_epochs} epochs: {', '.join(name for name in step['datasets'])}")
      full_datasets = []
      for dataset in step['datasets']:
        dataset_info = step_def['datasets'][dataset]
        full_datasets.append(get_dataset(dataset_info['name'],
                                         *dataset_info.get('args', ()),
                                         seed=seed,
                                         save_local=save_dataset,
                                         **dataset_info.get('kwargs', dict())))

      train = interleave_datasets([dataset['train'] for dataset in full_datasets])
      validation = interleave_datasets([dataset['validation'] for dataset in full_datasets])
      print(f"Loaded")
      yield step_name, step_epochs, train, validation
