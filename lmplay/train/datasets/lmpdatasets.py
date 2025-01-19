from .wikipedia import get_wikipedia
from .openorca import get_openorca
from .tinystories import get_tinystories


_DATASET_FUNCS = {'wiki':get_wikipedia,
                  'openorca':get_openorca,
                  'tinystories':get_tinystories}

def get_dataset(dataset_name:str, *args, seed=0, val_split=0.1, save_local:bool = False, **kwargs):
  if dataset_name in dataset_name:
    return _DATASET_FUNCS[dataset_name](*args, seed=seed, val_split=val_split, save_local=save_local, **kwargs)
  raise ValueError(f"Unknown dataset {dataset_name}")