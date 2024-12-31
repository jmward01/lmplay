from datasets import load_dataset, load_from_disk
import os
DEFAULT_DATASET_DIR = "out_gpt/datasets"
LMP_DATASETS_ENV = "LMP_DATASETS"

def _get_wikipedia(seed:int, val_split:float, date="20231101", lang="en", save_local:bool = False):
  """This will look for the dataset in the path indicated by LMP_DATASETS first and then grab it from huggingface if not found.
  if LMP_DATASETS is not set then  out_gpt/datasets will be assumed.

  :param seed: sed to randomize on.
  :param val_split: pct to dedicate to validation
  :param date: the dataset date in the hf repo
  :param lang: what lang to get. Refer to the hf repo for valid values
  :param save_local: save the dataset to LMP_DATASETS directory
  :return:
  """
  dataset_path = os.path.expanduser(os.environ.get(LMP_DATASETS_ENV, DEFAULT_DATASET_DIR))
  save_name = os.path.expanduser(os.path.join(dataset_path, f"wikimedia_wikipedia_{date}_{lang}.hf"))
  if not os.path.exists(save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("wikimedia/wikipedia", f"{date}.{lang}")['train']
    if save_local:
      os.makedirs(dataset_path, exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  return {'train':datasets['train'], 'validation':datasets['test']}


def get_wikipedia(lang:str, seed=0, val_split=0.1, save_local:bool = False):
  return _get_wikipedia(seed, val_split, lang=lang, save_local=save_local)
