"""
OpenOrca dataset loader for instruction following training.

This module handles loading and processing the Open-Orca/OpenOrca dataset from
HuggingFace Hub. OpenOrca is a large-scale instruction following dataset that
contains system prompts, questions, and responses suitable for fine-tuning
language models on instruction following tasks.

The dataset is processed to standardize the format:
- 'prompt': Combined system prompt and question
- 'truth': The target response

The module handles local caching to avoid repeated downloads.
"""

from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _map(example):
  """
  Transform OpenOrca dataset examples to lmplay format.
  
  Combines the system_prompt and question into a single prompt field,
  and maps the response to the truth field.
  
  Args:
    example: Single example from the OpenOrca dataset
    
  Returns:
    Transformed example with 'prompt' and 'truth' fields
  """
  example['prompt'] = f"{example['system_prompt']}\n{example['question']}"
  example['truth'] = example['response']
  return example




def _get_openorca(seed: int, val_split: float, save_local: bool = True):
  """
  Internal function to load and process the OpenOrca dataset.
  
  Loads the Open-Orca/OpenOrca dataset from HuggingFace Hub, processes it
  to the standard lmplay format, and optionally caches it locally.
  
  Args:
    seed: Random seed for reproducible train/validation split
    val_split: Fraction of data to use for validation (0.0 to 1.0)
    save_local: Whether to save processed dataset locally for faster reloading
    
  Returns:
    Dictionary with 'train' and 'validation' datasets
  """
  save_name = os.path.expanduser(os.path.join(dataset_path(), f"Open_Orca_OpenOrca"))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"

  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("Open-Orca/OpenOrca")['train']
    ds = ds.map(_map, remove_columns=ds.column_names)
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


def get_openorca(seed=0, val_split=0.1, save_local: bool = True):
  """
  Load the OpenOrca instruction following dataset.
  
  The OpenOrca dataset contains system prompts, questions, and responses
  suitable for training models on instruction following tasks. This function
  handles downloading, processing, and caching the dataset.
  
  Args:
    seed: Random seed for train/validation split (default: 0)
    val_split: Validation split fraction (default: 0.1)
    save_local: Save dataset locally for faster subsequent loading (default: True)
    
  Returns:
    Dictionary containing 'train' and 'validation' HuggingFace Dataset objects
    
  Example:
    dataset = get_openorca(seed=42, val_split=0.15)
    train_data = dataset['train']
    val_data = dataset['validation']
  """
  return _get_openorca(seed, val_split, save_local=save_local)
