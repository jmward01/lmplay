from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _get_wikipedia(seed:int, val_split:float, date="20231101", lang="en", save_local:bool = True):
  """This will look for the dataset in the path indicated by LMP_DATASETS first and then grab it from huggingface if not found.
  if LMP_DATASETS is not set then  out_gpt/datasets will be assumed.

  :param seed: sed to randomize on.
  :param val_split: pct to dedicate to validation
  :param date: the dataset date in the hf repo
  :param lang: what lang to get. Refer to the hf repo for valid values
  :param save_local: save the dataset to LMP_DATASETS directory
  :return:
  """

  save_name = os.path.expanduser(os.path.join(dataset_path(), f"wikimedia_wikipedia_{date}_{lang}"))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"
  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("wikimedia/wikipedia", f"{date}.{lang}")['train']
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


def get_wikipedia(lang:str, seed=0, val_split=0.1, save_local:bool = True):
  return _get_wikipedia(seed, val_split, lang=lang, save_local=save_local)