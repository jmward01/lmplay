from typing import Optional
from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def orca_transform(examples):
  prompts = []
  truths = []
  for system_prompt, question, response in zip(examples['system_prompt'], examples['question'], examples['response']):
    prompts.append(f"{system_prompt}\n{question}")
    truths.append(response)
  examples['prompt'] = prompts
  examples['truth'] = truths
  return examples




#Open-Orca/OpenOrca
def _get_openorca(seed:int, val_split:float, save_local:bool = False):
  save_name = os.path.expanduser(os.path.join(dataset_path(), f"Open_Orca_OpenOrca.hf"))
  if not os.path.exists(save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("Open-Orca/OpenOrca")['train']
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  datasets['train'].set_transform(orca_transform)
  datasets['test'].set_transform(orca_transform)
  return {'train':datasets['train'], 'validation':datasets['test']}

def get_openorca(seed=0, val_split=0.1, save_local:bool = False):
  return _get_openorca(seed, val_split, save_local=save_local)
