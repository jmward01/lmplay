import os
from typing import Optional

DEFAULT_DATASET_DIR = "out_gpt/datasets"
LMP_DATASETS_ENV = "LMP_DATASETS"

def dataset_path():
  return os.path.expanduser(os.environ.get(LMP_DATASETS_ENV, DEFAULT_DATASET_DIR))




def to_row(row: dict, max_length:int=None) -> (dict, Optional[int]):
  #user = f'Write an article with the following title: {row["title"]}'
  #system = 'You write wikipedia articles.'
  #truth = f'Title: {row["title"]}. Article: {row["text"]}'
  prompt = ""
  #return {'system': system, 'user': user, 'truth': truth}
  if 'prompt' in row and 'truth' in row:
    return {'prompt':row['prompt'], 'truth':row['truth']}, None
  elif 'truth' in row:
    return {'prompt':"", 'truth':row['truth']}, None
  if 'title' in row:
    prompt = f'Title: {row["title"]}\n'
  else:
    prompt = ''
  truth:str = row['text']
  ts = truth.split()
  ps = prompt.split()
  estimated_tokens = (len(ps) + len(ts))*1.5
  continuation = None
  if not max_length is None and estimated_tokens > max_length:
    #Find the number of words into the truth (approximately) to start looking for a split.
    start_trim = int(max_length/1.4 - len(ps)*1.5)
    #Now convert that ti an actual place in the string
    start_trim = len(' '.join(ts[:start_trim]))
    nearest_period = truth.find('. ', start_trim)
    if nearest_period > 0 and nearest_period  + 2 < len(row['text']):
      continuation = nearest_period + 2
  return {'prompt':prompt, 'truth': truth}, continuation

def continue_row(row:dict, continuation:int, max_length:int=None) -> (dict, Optional[int]):
  next_nearest_period = row['text'].find('. ', continuation)
  if 'title' in row:
    prompt = f'Title: {row["title"]}\n...'
  else:
    prompt = '...'
  if next_nearest_period > 0 and next_nearest_period + 2 < len(row['text']):
    prompt = prompt + row['text'][continuation:next_nearest_period + 2]
    continuation = next_nearest_period + 2

  truth = row['text'][continuation:]
  ts = truth.split()
  ps = prompt.split()
  estimated_tokens = (len(ps) + len(ts))*1.5
  new_continuation = None
  if not max_length is None and estimated_tokens > max_length:
    #Find the number of words into the truth (approximately) to start looking for a split.
    start_trim = int(max_length/1.4 - len(ps)*1.5)
    #Now convert that ti an actual place in the string
    start_trim = len(' '.join(ts[:start_trim]))
    nearest_period = truth.find('. ', start_trim)
    if nearest_period > 0 and nearest_period + continuation + 2 < len(row['text']):
      new_continuation = nearest_period + continuation + 2
  return {'prompt':prompt, 'truth': truth}, new_continuation


def batcher(dataset,
            batch_size: int,
            default_to_row=to_row,
            default_continue_row=continue_row,
            epochs:Optional[float]=None,
            allow_short=True,
            fast_forward:int=0,
            max_length:int=None):
  """Why build things this way? The 'standard' training that just concatenates strings together never really made sense to me.
  This will take each example and if it is too long, break it and continue it as another sample in the batch.
  If an example would go to a new batch it will be dropped since we don't want to cheat and let a new batch learn from previous batches.

  :param dataset: dataset to batch from
  :param batch_size: guess
  :param row_reader: converts rows from the dataset into a standard format for the model/continuation function.
  :param row_continuation: Function to create a continuation from a sample that is too long.
  :param epochs: I hate epochs but they are easy to think about. Curriculum is a different exp
  :param allow_short: let the final batch be short.
  :param fast_forward: For model re-starts
  :return:
  """
  batch = []
  count = 0
  dataset_len = len(dataset)
  running = True
  while running:

    if count == 0 and fast_forward > 0:
      offset = fast_forward % dataset_len
      count = fast_forward
    else:
      offset = 0
    #Just re-start the dataset
    new_count = 0
    while offset < dataset_len:
      row = dataset[offset]
      continuation = -1
      while continuation is not None:
        if continuation == -1:
          built_row, continuation = row.get('reader', default_to_row)(row, max_length=max_length)
          count += 1
          offset += 1
          new_count += 1
        else:
          #Gives a chance to 'continue' a row that was too long as another example.
          built_row, continuation = row.get('continuation',default_continue_row)(row, continuation, max_length=max_length)
        batch.append(built_row)
        if epochs is not None and count/len(dataset) >= epochs:
          #Looks like we have see enough examples.
          running = False
          break
        if len(batch) == batch_size:
          yield batch, new_count
          new_count = 0
          #Even if we have more we could get from this sample we want to break continuation for two reasons:
          #1) When a model re-starts this should allow batches to align making stats comparisons a little less fuzzy (not huge but nice)
          #2) We don't want to give the model a chance to 'cheat' by getting an update mid sample.
          # If this is desired behavior then it is a curriculum thing that should be built into the data loading.
          continuation = None
          batch = []


    if len(batch) > 0 and (allow_short or len(batch) == batch_size):
      yield batch, new_count