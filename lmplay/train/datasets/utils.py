"""
Dataset processing utilities for lmplay training.

This module provides utilities for dataset path management and batch processing
with intelligent text continuation. Key features:

- Configurable dataset storage paths via environment variables
- Smart text chunking that preserves sentence boundaries
- Batch processing with automatic continuation of long texts
- Support for both language modeling and instruction following formats

The batching system is designed to handle variable-length texts efficiently
while maintaining training stability by avoiding cross-batch continuations.
"""

import os
from typing import Optional

# Configuration constants
DEFAULT_DATASET_DIR = "out_gpt/datasets"  # Default dataset storage directory
LMP_DATASETS_ENV = "LMP_DATASETS"  # Environment variable for custom dataset path

def dataset_path():
  """
  Get the path for dataset storage.
  
  Returns the dataset storage path, either from the LMP_DATASETS environment
  variable or the default directory.
  
  Returns:
    Expanded path to the dataset storage directory
  """
  return os.path.expanduser(os.environ.get(LMP_DATASETS_ENV, DEFAULT_DATASET_DIR))




def to_row(row: dict, max_length: int = None) -> (dict, Optional[int]):
  """
  Convert dataset row to standard lmplay format with intelligent chunking.
  
  Handles both instruction following format (prompt/truth) and language modeling
  format (title/text). For long texts, finds appropriate split points at sentence
  boundaries to create continuable chunks.
  
  Args:
    row: Dataset row with 'prompt'/'truth' or 'title'/'text' fields
    max_length: Maximum estimated token length for output (uses ~1.5 tokens per word)
    
  Returns:
    Tuple of (formatted_row, continuation_position)
    - formatted_row: Dict with 'prompt' and 'truth' fields
    - continuation_position: Character position for continuation, or None if complete
  """
  if 'truth' in row and not row['truth'] is None:
    if 'prompt' in row and not row['prompt'] is None:
      return {'prompt':row['prompt'], 'truth':row['truth']}, None
    return {'prompt':"", 'truth':row['truth']}, None

  if 'title' in row and row['title'] is not None:
    prompt = f'Title: {row["title"]}\n'
  else:
    prompt = ''
  truth:str = row['text']
  ts = truth.split()
  ps = prompt.split()
  #This is the total estimated tokens
  estimated_tokens = (len(ps) + len(ts))*1.5
  continuation = None
  if not max_length is None and estimated_tokens > max_length:
    #we probably have an example that is too long
    #Find the number of words into the truth (approximately) to start looking for a split.
    start_trim = max(int(max_length/1.4 - len(ps)*1.5), 1)
    #Now convert that ti an actual place in the string
    start_trim = len(' '.join(ts[:start_trim]))
    nearest_period = truth.find('. ', start_trim)
    if nearest_period > 0 and nearest_period  + 2 < len(row['text']):
      continuation = nearest_period + 2
  return {'prompt':prompt, 'truth': truth}, continuation

def continue_row(row: dict, continuation: int, max_length: int = None) -> (dict, Optional[int]):
  """
  Create a continuation of a long text starting from a specific position.
  
  Generates a new training example that continues from where the previous chunk
  ended, with appropriate context and continuation markers.
  
  Args:
    row: Original dataset row
    continuation: Character position to continue from
    max_length: Maximum estimated token length for output
    
  Returns:
    Tuple of (formatted_row, next_continuation_position)
    - formatted_row: Dict with continuation prompt and remaining truth text
    - next_continuation_position: Position for next continuation, or None if complete
  """
  next_nearest_period = row['text'].find('. ', continuation)
  if 'title' in row and row['title'] is not None:
    prompt = f'Title: {row["title"]}\n...'
  else:
    prompt = '...'
  if next_nearest_period > 0 and next_nearest_period + 2 < len(row['text']):
    #trying to give it some runup in the form of a fragment of the last sentence as part of the prompt.
    prompt = prompt + row['text'][continuation:next_nearest_period + 2]
    continuation = next_nearest_period + 2

  truth = row['text'][continuation:]
  ts = truth.split()
  ps = prompt.split()
  estimated_tokens = (len(ps) + len(ts))*1.5
  new_continuation = None
  if not max_length is None and estimated_tokens > max_length:
    #Find the number of words into the truth (approximately) to start looking for a split.
    start_trim = max(int(max_length/1.4 - len(ps)*1.5), 1)
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
            epochs: Optional[float] = None,
            allow_short=True,
            fast_forward: int = 0,
            max_length: int = None):
  """
  Create batches from dataset with intelligent text continuation.
  
  This batching approach differs from standard concatenation by preserving
  example boundaries while handling long texts through continuation. Long
  examples are split at sentence boundaries and continued in the same batch,
  but continuations never cross batch boundaries to prevent information leakage.
  
  Args:
    dataset: HuggingFace dataset to batch
    batch_size: Number of examples per batch
    default_to_row: Function to convert dataset rows to standard format
    default_continue_row: Function to create continuations of long rows
    epochs: Number of epochs to run (can be fractional)
    allow_short: Allow final batch to be smaller than batch_size
    fast_forward: Number of samples to skip (for resuming training)
    max_length: Maximum token length for examples (triggers continuation)
    
  Yields:
    Tuple of (batch, new_samples_count)
    - batch: List of formatted examples with 'prompt' and 'truth' fields
    - new_samples_count: Number of new samples processed in this batch
    
  Notes:
    - Continuations within a batch maintain training stability
    - Cross-batch continuations are avoided to prevent information leakage
    - Custom row readers can be specified per dataset row
    - Token estimation uses ~1.5 tokens per word heuristic
  """
  batch = []
  #Total count of samples used accross all epochs
  count = 0
  dataset_len = len(dataset)
  running = True
  while running:

    if count == 0 and fast_forward > 0:
      #Offset is the place in the current epoch.
      offset = fast_forward % dataset_len
      count = fast_forward
    else:
      offset = 0
    #Just re-start the dataset
    new_count = 0
    while offset < dataset_len and running:
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