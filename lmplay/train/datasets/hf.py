from datasets import load_dataset, load_from_disk
from .utils import dataset_path
from functools import partial
import os,json


def _transform(examples, prompt_column=None, truth_column=None):
  if not prompt_column is None:
    prompts = []
    truths = []
    for prompt, truth in zip(examples[prompt_column], examples[truth_column]):
      if not isinstance(prompt, str):
        prompt = json.dumps(prompt)
      if not isinstance(truth, str):
        truth = json.dumps(truth)
      prompts.append(prompt)
      truths.append(truth)
    examples['prompt'] = prompts
    examples['truth'] = truths
    return examples

  truths = []
  for truth in examples[truth_column]:
    if not isinstance(truth, str):
      truth = json.dumps(truth)
    truths.append(truth)
  examples['truth'] = truths
  return examples


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
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  transform = partial(_transform, prompt_column=prompt_column, truth_column=truth_column)
  datasets = ds.train_test_split(test_size=val_split, seed=seed)
  datasets['train'].set_transform(transform)
  datasets['test'].set_transform(transform)
  return {'train': datasets['train'], 'validation': datasets['test']}


def get_hf(dataset_save_name, *arg, seed=0, val_split=0.1, save_local: bool = False, **kwarg):
  return _get_hf(seed, val_split, dataset_save_name, *arg, save_local=save_local, **kwarg)
