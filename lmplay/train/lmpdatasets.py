from datasets import load_dataset

def _get_wikipedia(name:str, seed:int, val_split:float, date="20220301"):
  #~184k articles
  train = load_dataset("wikipedia", f"{date}.{name}", split="train", beam_runner='DirectRunner')
  datasets = train.train_test_split(test_size=val_split, seed = seed)
  return {'train':datasets['train'], 'validation':datasets['test']}


def get_wikipedia_en(seed=0, val_split=0.1):
  return _get_wikipedia('en', seed, val_split)

def get_wikipedia_simple(seed=0, val_split=0.1):
  return _get_wikipedia('simple', seed, val_split)