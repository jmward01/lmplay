from typing import Optional
from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _map(example):
  example['prompt'] = f"{example['system_prompt']}\n{example['question']}"
  example['truth'] = example['response']
  return example




#Open-Orca/OpenOrca
def _get_openorca(seed:int, val_split:float, save_local:bool = False):
  save_name = os.path.expanduser(os.path.join(dataset_path(), f"Open_Orca_OpenOrca.hf"))
  if not os.path.exists(save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("Open-Orca/OpenOrca")['train']
    ds = ds.map(_map, remove_columns=ds.column_names)
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  return {'train':datasets['train'], 'validation':datasets['test']}

def get_openorca(seed=0, val_split=0.1, save_local:bool = False):
  return _get_openorca(seed, val_split, save_local=save_local)
