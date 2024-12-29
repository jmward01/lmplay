from datasets import load_dataset

def _get_wikipedia(seed:int, val_split:float, date="20231101", lang="en"):
  ds = load_dataset("wikimedia/wikipedia", f"{date}.{lang}")['train']
  datasets = ds.train_test_split(test_size=val_split, seed = seed)
  return {'train':datasets['train'], 'validation':datasets['test']}


def get_wikipedia(lang:str, seed=0, val_split=0.1):
  return _get_wikipedia(seed, val_split, lang=lang)
