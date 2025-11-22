"""
HuggingFace dataset loading utilities for lmplay training.

This module provides generic functionality for loading and processing datasets from
HuggingFace Hub. It handles:

- Automatic dataset download and caching
- Train/validation splits with reproducible seeds
- Column mapping and standardization to lmplay format
- Local dataset saving for offline use
- Cleanup of intermediate cache files

The module standardizes datasets to use 'prompt', 'truth', and 'text' columns
depending on the task type (instruction following vs. language modeling).
"""

from datasets import load_dataset, load_from_disk
from .utils import dataset_path
from functools import partial
import os, json

def _map(example, prompt_column=None, truth_column=None, text_column=None):
  """
  Map dataset columns to lmplay standard format.
  
  Converts dataset columns to the standard lmplay format:
  - 'text' column for language modeling tasks
  - 'prompt' and 'truth' columns for instruction following tasks
  
  Non-string values are converted to JSON strings.
  
  Args:
    example: Single dataset example
    prompt_column: Name of column containing prompts/questions
    truth_column: Name of column containing target responses/answers
    text_column: Name of column containing raw text for language modeling
    
  Returns:
    Modified example with standardized column names
  """
  if text_column is not None:
    t = example[text_column]
    if not isinstance(t, str):
      t = json.dumps(t)
    example['text'] = t
    return example
  truth = example[truth_column]
  if not isinstance(truth, str):
    truth = json.dumps(truth)
  example['truth'] = truth
  if not prompt_column is None:
    prompt = example[prompt_column]
    if not isinstance(prompt, str):
      prompt = json.dumps(prompt)
    example['prompt'] = prompt
    return example
  return example

def _get_hf(seed: int,
            val_split: float,
            dataset_save_name: str,
            *args,
            prompt_column: str = None,
            truth_column: str = None,
            text_column: str = None,
            save_local: bool = True,
            **kwargs):
  """
  Internal function to load and process HuggingFace datasets.
  
  Args:
    seed: Random seed for reproducible train/validation splits
    val_split: Fraction of data to use for validation (0.0 to 1.0)
    dataset_save_name: Local name for caching the processed dataset
    *args: Arguments passed to load_dataset()
    prompt_column: Column name containing prompts/questions
    truth_column: Column name containing target responses
    text_column: Column name containing raw text for language modeling
    save_local: Whether to save processed dataset locally
    **kwargs: Additional arguments passed to load_dataset()
    
  Returns:
    Dictionary with 'train' and 'validation' datasets
  """
  save_name = os.path.expanduser(os.path.join(dataset_path(), dataset_save_name))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"
  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset(*args, **kwargs, split='train')
    ds.cleanup_cache_files()
    map_fn = partial(_map, prompt_column=prompt_column, truth_column=truth_column, text_column=text_column)
    preserve_columns = {prompt_column, truth_column, text_column}
    to_remove = list(set(ds.column_names) - preserve_columns)
    ds_mapped = ds.map(map_fn, remove_columns=to_remove)
    ds.cleanup_cache_files()
    datasets = ds_mapped.train_test_split(test_size=val_split, seed=seed)
    ds_mapped.cleanup_cache_files()
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
  #transform = partial(_transform, prompt_column=prompt_column, truth_column=truth_column)

  return {'train': train, 'validation': validation}


def get_hf(dataset_save_name, *arg, seed=0, val_split=0.1, save_local: bool = True, **kwarg):
  """
  Load and process a HuggingFace dataset for lmplay training.
  
  This function loads datasets from HuggingFace Hub, processes them into the
  standard lmplay format, and optionally caches them locally for faster
  subsequent loading.
  
  Args:
    dataset_save_name: Local name for the processed dataset
    *arg: Arguments passed to load_dataset() (dataset name, config, etc.)
    seed: Random seed for train/validation split (default: 0)
    val_split: Validation split fraction (default: 0.1)
    save_local: Save processed dataset locally (default: True)
    **kwarg: Additional arguments (prompt_column, truth_column, text_column, etc.)
    
  Returns:
    Dictionary containing 'train' and 'validation' HuggingFace Dataset objects
    
  Example:
    # For instruction following dataset
    dataset = get_hf('my_dataset', 'org/dataset-name', 
                     prompt_column='question', truth_column='answer')
    
    # For language modeling dataset  
    dataset = get_hf('my_dataset', 'org/dataset-name',
                     text_column='content')
  """
  return _get_hf(seed, val_split, dataset_save_name, *arg, save_local=save_local, **kwarg)
