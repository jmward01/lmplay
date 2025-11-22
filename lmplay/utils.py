"""
Utility functions for the lmplay framework.

This module provides a collection of utility functions used throughout the
lmplay framework for various common operations including:

- Parameter naming and serialization for experiment tracking
- Mask generation for causal attention
- Factory functions for creating linear layers with optional purpose tracking
- Default value handling and decorator utilities
- Tokenizer initialization with proper caching setup

These utilities support the framework's emphasis on experimental reproducibility
and modular architecture design.
"""

import torch
import os
import tiktoken

__all__ = ['gen_mask', 'create_linear', 'accepts_purpose', 'set_accepts_purpose', 'to_name', 'pstr', 'get_tokenizer']


def pstr(v) -> str:
  """Convert parameter values to name-friendly strings for experiment tracking.
  
  This function serializes various Python objects into short string representations
  suitable for use in file names and experiment identifiers. It handles common
  parameter types used in machine learning experiments.
  
  Args:
    v: Value to convert. Can be None, DEFAULT, str, float, dict, iterable, or numeric.
  
  Returns:
    str: Short string representation:
      - None -> 'N'
      - DEFAULT -> 'D' 
      - str -> unchanged
      - float -> formatted to 1 decimal place
      - dict -> concatenated string values with '_' prefix
      - iterable -> concatenated string values with '_' prefix
      - numeric -> integer string
  
  Example:
    pstr(0.5) -> "0.5"
    pstr([1, 2, 3]) -> "_123"
    pstr(None) -> "N"
  """
  if v is None:
    return 'N'
  if v is DEFAULT:
    return 'D'
  if isinstance(v, str):
    return v
  if isinstance(v, float):
    return f"{v:0.1f}"
  if isinstance(v, dict):
    return f"_{''.join(pstr(vc) for vc in v.values())}"
  if hasattr(v, "__iter__"):
    return f"_{''.join(pstr(vc) for vc in v)}"
  return str(int(v))

def to_name(version:str, *args, **kwargs):
  """Generate a unique name string from version and parameters.
  
  Creates a standardized name for experiments by combining a version string
  with serialized parameter values. This is used for generating unique
  identifiers for model runs, checkpoints, and experiment tracking.
  
  Args:
    version (str): Base version or experiment name.
    *args: Positional arguments to include in the name.
    **kwargs: Keyword arguments to include in the name.
  
  Returns:
    str: Combined name string in format: version_args_kwargs
  
  Example:
    to_name("exp1", 5, 0.1, lr=0.001) -> "exp1_50.1_0.001"
  """
  name = version
  if len(args) > 0:
    name = f"{name}_{''.join(pstr(v) for v in args)}"
  if len(kwargs) > 0:
    name = f"{name}_{'_'.join(pstr(v) for v in kwargs.values())}"
  return name

class _DEFAULT():
  """Sentinel class for representing default parameter values.
  
  This class is used as a marker for default values in function parameters,
  allowing the framework to distinguish between explicitly passed None values
  and parameters that should use their defaults.
  """
  pass

DEFAULT = _DEFAULT.__class__


def ignore_default(f:callable) -> callable:
  """Decorator that filters out DEFAULT values from keyword arguments.
  
  This decorator automatically removes any keyword arguments that have DEFAULT
  as their value before calling the wrapped function. This allows for cleaner
  parameter handling when using the DEFAULT sentinel value.
  
  Args:
    f (callable): Function to wrap.
  
  Returns:
    callable: Wrapped function that filters DEFAULT values from kwargs.
  
  Example:
    @ignore_default
    def func(a, b=None):
      return a, b
    
    func(1, b=DEFAULT)  # Calls func(1) with b getting its default None
  """
  def new_f(*args, **kwargs):
    kwargs = {k:v for k,v in kwargs.items() if not v is DEFAULT}
    return f(*args, **kwargs)
  return new_f

def gen_mask(max_len: int) -> torch.Tensor:
  """Generate a causal attention mask for autoregressive models.
  
  Creates a lower triangular mask that prevents attention to future positions
  in the sequence. This is essential for causal (autoregressive) language models
  where each position should only attend to previous positions.
  
  Args:
    max_len (int): Maximum sequence length for the mask.
  
  Returns:
    torch.Tensor: Boolean mask of shape (1, 1, max_len, max_len) where
      mask[i, j] = 1 if position i can attend to position j, 0 otherwise.
      Unsqueezed to work with multi-head attention (batch, heads, seq, seq).
  
  Example:
    gen_mask(3) creates a 3x3 mask:
    [[1, 0, 0],
     [1, 1, 0], 
     [1, 1, 1]]
  """
  return torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)


def accepts_purpose(o) -> bool:
  """Check if an object accepts a 'purpose' parameter.
  
  Tests whether an object (typically a class or function) has been marked
  as accepting a 'purpose' parameter for tracking the intended use of
  created instances. This is used in the factory pattern for linear layers.
  
  Args:
    o: Object to test for purpose parameter support.
  
  Returns:
    bool: True if the object accepts purpose parameters, False otherwise.
  """
  if not hasattr(o, 'accepts_purpose'):
    return False
  return o.accepts_purpose


def set_accepts_purpose(o, v: bool = True):
  """Mark an object as accepting purpose parameters.
  
  Sets the 'accepts_purpose' attribute on an object to indicate that it
  supports purpose tracking. This is used to mark classes that can accept
  purpose strings for identifying the role of created instances.
  
  Args:
    o: Object to mark as accepting purpose parameters.
    v (bool): Whether the object accepts purpose parameters. Defaults to True.
  
  Returns:
    The object with the accepts_purpose attribute set.
  
  Example:
    linear_class = set_accepts_purpose(MyLinear)
    # Now MyLinear will receive purpose parameters in create_linear()
  """
  if not hasattr(o, 'accepts_purpose'):
    setattr(o, 'accepts_purpose', v)
  return o


def create_linear(linear_class, purpose:str, *args, **kwargs):
  """Factory function for creating linear layers with optional purpose tracking.
  
  Creates a linear layer instance, optionally passing a purpose string to
  classes that support it. This allows for consistent creation of linear
  layers throughout the framework while supporting purpose tracking for
  debugging and analysis.
  
  Args:
    linear_class: Class or factory function for creating linear layers.
    purpose (str): String describing the intended use of this linear layer
      (e.g., "mha_query", "block_ff_1"). Only passed if the class accepts it.
    *args: Positional arguments passed to the linear class constructor.
    **kwargs: Keyword arguments passed to the linear class constructor.
  
  Returns:
    Instance of the linear layer class.
  
  Example:
    # For a class that accepts purpose:
    layer = create_linear(ULinear, "attention_proj", 512, 1024)
    # Calls: ULinear(512, 1024, purpose="attention_proj")
    
    # For a standard class:
    layer = create_linear(nn.Linear, "mlp_layer", 512, 1024) 
    # Calls: nn.Linear(512, 1024)
  """
  if accepts_purpose(linear_class):
    l = linear_class(*args, purpose=purpose, **kwargs)
  else:
    l = linear_class(*args, **kwargs)
  return l


def get_tokenizer(encoding_name: str = "gpt2"):
  """Get a tokenizer with proper cache directory configuration.

  This function ensures that tiktoken can cache tokenizer data locally.
  It checks the TIKTOKEN_CACHE_DIR environment variable and, if not set,
  configures it to use {output_dir}/tokenizer where output_dir is determined by:
  1. LMP_DATASETS environment variable if set
  2. ./out_gpt otherwise

  The cache directory is created if it doesn't exist. This avoids repeated
  downloads of tokenizer data and improves startup time.

  Args:
    encoding_name (str): Name of the tiktoken encoding to load.
      Defaults to "gpt2".

  Returns:
    tiktoken encoding object ready for use.

  Environment Variables:
    TIKTOKEN_CACHE_DIR: If set, used directly as the cache directory.
    LMP_DATASETS: Base output directory (falls back to ./out_gpt).

  Example:
    tokenizer = get_tokenizer("gpt2")
    tokens = tokenizer.encode("Hello, world!")
  """
  # Check if TIKTOKEN_CACHE_DIR is already set
  if "TIKTOKEN_CACHE_DIR" not in os.environ:
    # Determine the output directory
    if "LMP_DATASETS" in os.environ:
      output_dir = os.path.expanduser(os.environ["LMP_DATASETS"])
    else:
      output_dir = "./out_gpt"

    # Set the cache directory
    cache_dir = os.path.join(output_dir, "tokenizer")
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

    # Create the directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

  # Load and return the tokenizer
  return tiktoken.get_encoding(encoding_name)

