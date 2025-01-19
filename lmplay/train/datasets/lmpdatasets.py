from .wikipedia import get_wikipedia
from .openorca import get_openorca
from .orcamathword200 import get_orcamathword200
from .tinystories import get_tinystories
from .hf import get_hf
from .plan_configs import ALL_DATASETS

_DATASET_FUNCS = {'wiki': get_wikipedia,
                  'openorca': get_openorca,
                  'orcamathword200': get_orcamathword200,
                  'tinystories': get_tinystories,
                  'hf': get_hf}

def get_dataset(ds_loader: str, *args, seed=0, val_split=0.1, save_local: bool = False, **kwargs):
  if ds_loader in _DATASET_FUNCS:
    return _DATASET_FUNCS[ds_loader](*args, seed=seed, val_split=val_split, save_local=save_local, **kwargs)
  raise ValueError(f"Unknown dataset loader {ds_loader}")


def get_known_plan_dataset(dataset_name:str, save_local: bool = False):
  dataset = ALL_DATASETS[dataset_name]
  return get_dataset(dataset['ds_loader'], *dataset.get('args', ()), save_local=save_local, **dataset.get('kwargs', dict()))