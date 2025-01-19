from datasets import load_dataset, load_from_disk
from .utils import dataset_path
from functools import partial
import os,json

def _map(example, prompt_column=None, truth_column=None):
  truth = example[truth_column]
  if not isinstance(truth, str):
    truth = json.dumps(truth)
  example['truth'] = truth
  if not prompt_column is None:
    prompt = example[prompt_column]
    if not isinstance(prompt, str):
      prompt = json.dumps(prompt)
    example['prompt'] = prompt
    return example
  return example

def _get_hf(seed: int,
            val_split: float,
            dataset_save_name: str,
            *args,
            prompt_column: str = None,
            truth_column: str = None,
            save_local: bool = False,
            **kwargs):
  save_name = os.path.expanduser(os.path.join(dataset_path(), dataset_save_name))
  if not os.path.exists(save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset(*args, **kwargs)['train']
    map_fn = partial(_map, prompt_column=prompt_column, truth_column=truth_column)
    ds = ds.map(map_fn, remove_columns=ds.column_names)
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  #transform = partial(_transform, prompt_column=prompt_column, truth_column=truth_column)

  datasets = ds.train_test_split(test_size=val_split, seed=seed)
  train = datasets['train']
  validation = datasets['test']
  return {'train': train, 'validation': validation}


def get_hf(dataset_save_name, *arg, seed=0, val_split=0.1, save_local: bool = False, **kwarg):
  return _get_hf(seed, val_split, dataset_save_name, *arg, save_local=save_local, **kwarg)
