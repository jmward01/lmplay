"""
Dataset loading coordination for lmplay training.

This module provides a unified interface for loading various datasets used in
language model training. It coordinates between specialized loaders for:

- Wikipedia articles (multiple languages)
- OpenOrca instruction following dataset
- Generic HuggingFace datasets
- Predefined dataset configurations

The module acts as a registry and dispatcher, routing dataset requests to the
appropriate specialized loader based on the dataset type.
"""

from .wikipedia import get_wikipedia
from .openorca import get_openorca
from .hf import get_hf
from .plan_configs import ALL_DATASETS

# Registry mapping dataset loader names to their functions
_DATASET_FUNCS = {'wiki': get_wikipedia,
                  'openorca': get_openorca,
                  'hf': get_hf}

def get_dataset(ds_loader: str, *args, seed=0, val_split=0.1, save_local: bool = False, **kwargs):
  """
  Load a dataset using the specified loader.
  
  This function dispatches to the appropriate specialized dataset loader
  based on the ds_loader parameter.
  
  Args:
    ds_loader: Name of the dataset loader ('wiki', 'openorca', 'hf')
    *args: Arguments passed to the specific loader function
    seed: Random seed for train/validation split (default: 0)
    val_split: Validation split fraction (default: 0.1)
    save_local: Save dataset locally for faster subsequent loading
    **kwargs: Additional keyword arguments for the loader
    
  Returns:
    Dictionary with 'train' and 'validation' datasets
    
  Raises:
    ValueError: If ds_loader is not recognized
    
  Example:
    # Load Wikipedia dataset
    dataset = get_dataset('wiki', 'en', seed=42, val_split=0.1)
    
    # Load OpenOrca dataset
    dataset = get_dataset('openorca', seed=42)
    
    # Load HuggingFace dataset
    dataset = get_dataset('hf', 'dataset_name', 'org/repo',
                         prompt_column='question', truth_column='answer')
  """
  if ds_loader in _DATASET_FUNCS:
    return _DATASET_FUNCS[ds_loader](*args, seed=seed, val_split=val_split, save_local=save_local, **kwargs)
  raise ValueError(f"Unknown dataset loader {ds_loader}")


def get_known_plan_dataset(dataset_name: str, save_local: bool = False):
  """
  Load a predefined dataset by name from the plan configurations.
  
  This function loads datasets that are predefined in the ALL_DATASETS
  configuration, which includes common datasets with their loading
  parameters already specified.
  
  Args:
    dataset_name: Name of the dataset from ALL_DATASETS configuration
    save_local: Save dataset locally for faster subsequent loading
    
  Returns:
    Dictionary with 'train' and 'validation' datasets
    
  Raises:
    KeyError: If dataset_name is not found in ALL_DATASETS
    
  Example:
    # Load predefined Wikipedia English dataset
    dataset = get_known_plan_dataset('wiki_en')
    
    # Load predefined OpenOrca dataset
    dataset = get_known_plan_dataset('openorca')
  """
  dataset = ALL_DATASETS[dataset_name]
  return get_dataset(dataset['ds_loader'], *dataset.get('args', ()), save_local=save_local, **dataset.get('kwargs', dict()))