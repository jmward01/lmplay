from datasets import load_dataset, load_from_disk
import os
DATASET_DIR = "out_gpt/datasets"

def _get_wikipedia(seed:int, val_split:float, date="20231101", lang="en"):
  save_name = os.path.expanduser(f"{DATASET_DIR}/wikimedia_wikipedia_{date}_{lang}.hf")
  if not os.path.exists(save_name):
    print(f"{save_name} not found. Downloading from hf.")
    os.makedirs(DATASET_DIR, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", f"{date}.{lang}")['train']
    ds.save_to_disk(save_name)
    print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  return {'train':datasets['train'], 'validation':datasets['test']}


def get_wikipedia(lang:str, seed=0, val_split=0.1):
  return _get_wikipedia(seed, val_split, lang=lang)
