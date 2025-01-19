from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _transform(examples):
  prompts = []
  truths = []
  for question, answer in zip(examples['question'], examples['answer']):
    prompts.append(f"{question}\nAnswer:")
    truths.append(answer)
  examples['prompt'] = prompts
  examples['truth'] = truths
  return examples




#microsoft/orca-math-word-problems-200k
def _get_orcamathword200(seed:int, val_split:float, save_local:bool = False):
  save_name = os.path.expanduser(os.path.join(dataset_path(), f"orca_math_word_problems_200k.hf"))
  if not os.path.exists(save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("microsoft/orca-math-word-problems-200k")['train']
    if save_local:
      os.makedirs(dataset_path(), exist_ok=True)
      ds.save_to_disk(save_name)
      print(f"{save_name} saved")
  else:
    ds = load_from_disk(save_name)
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  datasets['train'].set_transform(_transform)
  datasets['test'].set_transform(_transform)
  return {'train':datasets['train'], 'validation':datasets['test']}

def get_orcamathword200(seed=0, val_split=0.1, save_local:bool = False):
  return _get_orcamathword200(seed, val_split, save_local=save_local)
