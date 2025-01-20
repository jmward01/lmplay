from datasets import load_dataset, load_from_disk
from .utils import dataset_path
from functools import partial
import os,json

def _map(example, prompt_column=None, truth_column=None, text_column=None):
  if text_column is not None:
    t = example[text_column]
    if not isinstance(t, str):
      t = json.dumps(t)
    example['text'] = t
    return example
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
            text_column:str = None,
            save_local: bool = True,
            **kwargs):
  save_name = os.path.expanduser(os.path.join(dataset_path(), dataset_save_name))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"
  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset(*args, **kwargs, split='train')
    ds.cleanup_cache_files()
    map_fn = partial(_map, prompt_column=prompt_column, truth_column=truth_column, text_column=text_column)
    preserve_columns = {prompt_column, truth_column, text_column}
    to_remove = list(set(ds.column_names) - preserve_columns)
    ds_mapped = ds.map(map_fn, remove_columns=to_remove)
    ds.cleanup_cache_files()
    datasets = ds_mapped.train_test_split(test_size=val_split, seed=seed)
    ds_mapped.cleanup_cache_files()
    train = datasets['train']
    validation = datasets['test']
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      train.save_to_disk(train_save_name)
      validation.save_to_disk(validation_save_name)
      print(f"{save_name} saved")

  else:
    train = load_from_disk(train_save_name)
    validation = load_from_disk(validation_save_name)
  #transform = partial(_transform, prompt_column=prompt_column, truth_column=truth_column)

  return {'train': train, 'validation': validation}


def get_hf(dataset_save_name, *arg, seed=0, val_split=0.1, save_local: bool = True, **kwarg):
  return _get_hf(seed, val_split, dataset_save_name, *arg, save_local=save_local, **kwarg)
