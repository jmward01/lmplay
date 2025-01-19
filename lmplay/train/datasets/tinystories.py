from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

#def tinystories_transform(examples):
#  prompts = []
#  truths = []
#  for system_prompt, question, response in zip(examples['system_prompt'], examples['question'], examples['response']):
#    prompts.append(f"{system_prompt}\n{question}")
#    truths.append(response)
#  examples['prompt'] = prompts
#  examples['truth'] = truths
#  return examples




#roneneldan/TinyStories
def _get_tinystories(seed:int, val_split:float, save_local:bool = False):
  train_save_name = os.path.expanduser(os.path.join(dataset_path(), f"TinyStories_train.hf"))
  validation_save_name = os.path.expanduser(os.path.join(dataset_path(), f"TinyStories_validation.hf"))
  if not os.path.exists(train_save_name):
    print(f"{train_save_name} not found locally. Downloading from hf.")
    ds = load_dataset("roneneldan/TinyStories")
    ds_train = ds['train'].shuffle(seed)
    ds_validation = ds['validation'].shuffle(seed)
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds_train.save_to_disk(train_save_name)
      print(f"{train_save_name} saved")
      ds_validation.save_to_disk(validation_save_name)
      print(f"{validation_save_name} saved")
  else:
    ds_train = load_from_disk(train_save_name)
    ds_validation = load_from_disk(validation_save_name)
  #ds_train.set_transform(tinystories_transform)
  #ds_validation.set_transform(tinystories_transform)
  return {'train':ds_train, 'validation':ds_validation}

def get_tinystories(seed=0, val_split=0.1, save_local:bool = False):
  return _get_tinystories(seed, val_split, save_local=save_local)


