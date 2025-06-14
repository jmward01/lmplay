"""
Wikipedia dataset loader for language model pretraining.

This module handles loading Wikipedia articles from HuggingFace Hub for
language model pretraining. It supports multiple languages and provides
automatic caching for faster subsequent loading.

Features:
- Multi-language Wikipedia support (English, Spanish, etc.)
- Automatic train/validation splitting with reproducible seeds
- Local caching to avoid repeated downloads
- Configurable validation split ratios

The Wikipedia datasets are used primarily for pretraining language models
on high-quality, encyclopedic text before fine-tuning on more specialized tasks.
"""

from datasets import load_dataset, load_from_disk
from .utils import dataset_path
import os

def _get_wikipedia(seed: int, val_split: float, date="20231101", lang="en", save_local: bool = True):
  """
  Internal function to load Wikipedia dataset for a specific language.
  
  Checks for locally cached dataset first, then downloads from HuggingFace Hub
  if not found. Automatically splits into training and validation sets.
  
  Args:
    seed: Random seed for reproducible train/validation split
    val_split: Fraction of data to use for validation (0.0 to 1.0)
    date: Wikipedia dump date (default: "20231101")
    lang: Language code ("en", "es", etc.)
    save_local: Whether to save processed dataset locally
    
  Returns:
    Dictionary with 'train' and 'validation' datasets
  """

  save_name = os.path.expanduser(os.path.join(dataset_path(), f"wikimedia_wikipedia_{date}_{lang}"))
  train_save_name = f"{save_name}.train"
  validation_save_name = f"{save_name}.validation"
  if not os.path.exists(train_save_name) or not os.path.exists(validation_save_name):
    print(f"{save_name} not found locally. Downloading from hf.")
    ds = load_dataset("wikimedia/wikipedia", f"{date}.{lang}")['train']
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


def get_wikipedia(lang: str, seed=0, val_split=0.1, save_local: bool = True):
  """
  Load Wikipedia dataset for the specified language.
  
  Downloads and processes Wikipedia articles for language model pretraining.
  The dataset includes article titles and full text content, suitable for
  general language understanding tasks.
  
  Args:
    lang: Language code ("en" for English, "es" for Spanish, etc.)
    seed: Random seed for train/validation split (default: 0)
    val_split: Validation split fraction (default: 0.1)
    save_local: Save dataset locally for faster subsequent loading (default: True)
    
  Returns:
    Dictionary containing 'train' and 'validation' HuggingFace Dataset objects
    
  Example:
    # Load English Wikipedia
    en_dataset = get_wikipedia('en', seed=42, val_split=0.15)
    
    # Load Spanish Wikipedia  
    es_dataset = get_wikipedia('es')
  """
  return _get_wikipedia(seed, val_split, lang=lang, save_local=save_local)