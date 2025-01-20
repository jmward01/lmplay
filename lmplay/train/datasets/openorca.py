from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _map(example):
  example['prompt'] = f"{example['system_prompt']}\n{example['question']}"
  example['truth'] = example['response']
  return example




#Open-Orca/OpenOrca
def _get_openorca(seed:int, val_split:float, save_local:bool = True):
  save_name = os.path.expanduser(os.path.join(dataset_path(), f"Open_Orca_OpenOrca"))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"

  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("Open-Orca/OpenOrca")['train']
    ds = ds.map(_map, remove_columns=ds.column_names)
    datasets = ds.train_test_split(test_size=val_split, seed=seed)
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
  return {'train':train, 'validation':validation}


def get_openorca(seed=0, val_split=0.1, save_local:bool = True):
  return _get_openorca(seed, val_split, save_local=save_local)
