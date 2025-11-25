"""Base model classes and training infrastructure for the lmplay framework.

This module provides the core abstractions for building and training neural network models,
particularly language models. It includes:

- MBase: The foundational model class that all models inherit from
- LMBase: Specialized base class for language models
- LMRunnerBase: Abstract base for model runners that handle training/inference
- BasicModelRunner: Simple implementation of a model runner
- OptimizerWarmupLRScheduler: Learning rate scheduler with warmup
- Helper utilities for gradient freezing and context management

The module implements a runner pattern where models are wrapped in runners that
manage the training loop, optimization, checkpointing, and statistics tracking.
"""

import logging
from abc import abstractmethod
from typing import Optional, Sequence, List
import torch
from torch import nn

from lmplay.base.utils import apply_temperature, apply_repetition_penalty, top_k_filtering, top_p_filtering
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from lmplay.utils import ignore_default

logger = logging.getLogger(__name__)

#Lot's here to think about. embed is hopefully obvious. low run tokens would slowly zero out
#bias should be allowed to go wherever it needs to.
#ln is a regularization mechanism in its own right
#everything in attn is a question though.
# From what I have read weight decay in attn leads to poor performance.
# Gimme a bunch of GPUs and we can test that but for now just exclude it.
DEFAULT_WEIGHT_DECAY_EXCLUSION_PATTERNS = (".bias", "_bias", ".ln", "_ln", "ln.", "embed", "attn")

class MBase(nn.Module):
  """Base class for all models in the lmplay framework.
  
  This class provides core functionality for model initialization, tokenization,
  training, generation, and device management. All models should inherit from
  this class or one of its subclasses.
  
  Attributes:
    name: Model name for identification and file saving
    max_len: Maximum sequence length the model can handle
    expect_extra_loss: Whether the model returns additional loss terms
    pass_lengths: Whether to pass sequence lengths to the forward method
    flat_batch: Whether to flatten batches for processing
    tokenizer: Tokenizer instance for text processing
  """


  @ignore_default
  def __init__(self,
               name: str,
               *init_args,
               expect_extra_loss=False,
               pass_lengths=False,
               flat_batch=False,
               weight_decay_exclusion_patterns = DEFAULT_WEIGHT_DECAY_EXCLUSION_PATTERNS,
               max_len = 1024,
               **init_kwargs):
    super().__init__()
    init_kwargs['max_len'] = max_len
    init_kwargs['weight_decay_exclusion_patterns'] = weight_decay_exclusion_patterns
    self.weight_decay_exclusion_patterns = weight_decay_exclusion_patterns
    self.name = name.replace('.', '_')
    self.init_args = init_args
    self.init_kwargs = init_kwargs
    self.max_len = max_len
    self.expect_extra_loss = expect_extra_loss
    self.pass_lengths = pass_lengths
    self.flat_batch = flat_batch
    self._cached_device = None

  @property
  def device(self):
    """Get the device of the model, with caching for performance.
    
    Returns:
      torch.device: The device where the model parameters are located
    """
    if self._cached_device is None:
      # Use fc.weight.device as the reference for the model's device
      self._cached_device = self.fc.weight.device
    return self._cached_device

  def initialize(self, device):
    """Initialize model components after moving to device.
    
    This method can be overridden by subclasses to perform device-specific
    initialization.
    
    Args:
      device: The device to initialize on
    """
    pass
    # self.unified_tok_embed.initialize(self.tok_embed, device)

  def _kv_cache(self, cache: Optional[list], idx):
    """Get or create cache entry for key-value caching.
    
    Args:
      cache: Optional list of cached values
      idx: Index for the cache entry
      
    Returns:
      The cache entry at the specified index, or None if cache is None
    """
    if cache is None:
      return None
    if len(cache) <= idx:
      cache.append([])
    return cache[idx]

  def _tokenize_str(self, sample: dict, device, trim=True) -> (torch.Tensor, int):
    """Tokenize a single sample into tensor format.
    
    Args:
      sample: Dictionary with 'prompt' and optionally 'truth' keys
      device: Device to place the tensor on
      trim: Whether to trim sequences that exceed max_len
      
    Returns:
      tuple: (tokens tensor, prediction_starts index)
    """
    prompt = sample['prompt']
    tokens = [self.tokenizer.eot_token] + self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    # adjusting because the the model is - 1 on its prediction.
    # for example if the model sees this (& = bos/eos):
    # & P R E D I C
    # It will predict like this:
    # P R E D I C T
    # So it we said the prompt ended on 1 then prediction starts on 0
    prediction_starts = len(tokens) - 1
    if 'truth' in sample:
      truth: str = sample['truth']
      if not truth.endswith("<|endoftext|>"):
        tokens.extend(self.tokenizer.encode(sample['truth'] + "<|endoftext|>", allowed_special={"<|endoftext|>"}))
      else:
        tokens.extend(self.tokenizer.encode(sample['truth'], allowed_special={"<|endoftext|>"}))

    # We can go one more because one is being trimmed off
    if trim and len(tokens) > self.max_len + 1:
      # too long. Just cut it off
      tokens = tokens[:self.max_len + 1]
      # tokens[-1] = self.tokenizer.eot_token
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    return tokens, prediction_starts

  def _tokenize_batch(self, batch: Sequence[dict], dont_pad=False) -> (torch.Tensor, Sequence[int]):
    """Tokenize a batch of samples and optionally pad them.
    
    Args:
      batch: Sequence of sample dictionaries
      dont_pad: If True, skip padding (used for flat_batch mode)
      
    Returns:
      tuple: (padded tokens tensor, prediction_starts list, prediction_ends list)
    """
    device = self.device  # Use cached device property
    predictions_starts = []
    predictions_ends = []
    x = []
    for t in batch:
      t, ps = self._tokenize_str(t, device)
      x.append(t)
      predictions_starts.append(ps)
      predictions_ends.append(int(t.size(-1)) - 1)
    if not dont_pad:
      x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.eot_token)
    return x, predictions_starts, predictions_ends

  def to(self, *args, **kwargs):
    """Move model to specified device/dtype, with cache invalidation.
    
    Overrides PyTorch's to() method to ensure cached device is cleared
    when the model is moved.
    
    Returns:
      Self for method chaining
    """
    # Modified from pytorch 2.1 source
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    
    # Clear cached device when model is moved
    if device is not None:
      self._cached_device = None

    def convert(t):
      if convert_to_format is not None and t.dim() in (4, 5):
        if hasattr(t, 'force_device'):
          return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None,
                      non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking, memory_format=convert_to_format)
      if hasattr(t, 'force_device'):
        return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
      return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)

  @abstractmethod
  def forward(self, *args, **kwargs):
    """Forward pass of the model. Must be implemented by subclasses.
    
    Returns:
      Model outputs (specific format depends on the model type)
    """
    pass

  def train_prompts(self, prompts: Sequence[dict], include_prompts=True) -> (Sequence[str], torch.Tensor):
    """Process prompts for training, computing loss and predictions.
    
    Args:
      prompts: Sequence of prompt dictionaries with 'prompt' and 'truth' keys
      include_prompts: Whether to include prompt tokens in loss calculation
      
    Returns:
      tuple: (predictions list, total loss, total token count)
    """
    # We want to pad them together so that the truths will line up with the prompts.
    x, predictions_starts, predictions_ends = self._tokenize_batch(prompts, dont_pad=self.flat_batch)
    if self.flat_batch:
      truths = torch.concat([t[1:] for t in x], dim = 0)
      x = torch.concat([t[:-1] for t in x], dim = 0)
    else:
      # Truth doesn't have the first EOT char. It needs to start on prediction start
      truths = x[:, 1:]
      # x doesn't need the last EOT since it will be predicting that
      x = x[:, :-1]

    if self.pass_lengths:
      #These ends should already be one short/match the actual prediction end
      x = self(x, lengths=torch.tensor(predictions_ends, dtype=torch.int64, device=x.device))
    else:
      x = self(x)

    if self.expect_extra_loss:
      x, extra_loss = x
    else:
      extra_loss = None
    results = []
    if self.flat_batch:
      # num classes is always second. For, reasons?
      target_loss = F.cross_entropy(x.unsqueeze(0).permute(0, 2, 1), truths.unsqueeze(0), reduction="none").squeeze(0)

      if not extra_loss is None and extra_loss.shape[0] == target_loss.shape[0]:
        target_loss = target_loss + extra_loss
        extra_loss = None

      target_loss = torch.split(target_loss, predictions_ends)
    else:
      target_loss = F.cross_entropy(x.permute(0, 2, 1), truths, reduction="none")

    # num classes is always second. For, reasons?
    if extra_loss is None:
      total_target_loss = 0.0
    else:
      total_target_loss = torch.sum(extra_loss)
    total_token_count = 0
    for tl, prediction_start, prediction_end in zip(target_loss, predictions_starts, predictions_ends):
      if not include_prompts:
        tl = tl[prediction_start:prediction_end]
      else:
        tl = tl[:prediction_end]
        prediction_start = 0
      tl = tl.sum()
      token_count = max(prediction_end - prediction_start, 1)
      total_token_count += token_count
      # norm by number of tokens in the truth
      # total_target_loss = tl / token_count + total_target_loss
      total_target_loss = total_target_loss + tl

    # Get the predicted value so we can cut just that out as the result.
    predicted_tokens = torch.argmax(x, dim=-1)
    # for result, prediction_start, prediction_end, truth in zip(predicted_tokens, predictions_starts, predictions_ends, truths):
    if self.flat_batch:
      for result in torch.split(predicted_tokens, predictions_ends):
        # we only care about the prediction.
        # the last value is end of sentence
        results.append(self.tokenizer.decode(result.tolist()))
    else:

      for result, prediction_start, prediction_end in zip(predicted_tokens, predictions_starts, predictions_ends):
        # we only care about the prediction.
        # the last value is end of sentence
        results.append(self.tokenizer.decode(result[prediction_start:prediction_end].tolist()))
    # target loss is normalized by example but not by batch. That will be done by the caller.
    return results, total_target_loss, total_token_count

  def generate_prompts(self,
                       prompts: Sequence[dict],
                       max_len: Optional[int] = None,
                       temperature: float = 1.0,
                       top_k: Optional[int] = None,
                       top_p: float = 1.0,
                       repetition_penalty: float = 1.0,
                       do_sample: bool = False) -> Sequence[str]:
    """Generate text completions for given prompts with sampling options.

    Args:
      prompts: Sequence of prompt dictionaries with 'prompt' key
      max_len: Maximum generation length (uses model max_len if None)
      temperature: Sampling temperature. < 1.0 = sharper distribution (more deterministic),
                   > 1.0 = softer distribution (more random). 1.0 = no change.
      top_k: Keep only top k highest probability tokens. None/0 = disabled.
      top_p: Nucleus sampling - keep tokens with cumulative prob <= top_p. 1.0 = disabled.
      repetition_penalty: Penalty for repeated tokens. > 1.0 discourages repetition.
      do_sample: If True, use sampling. If False, use greedy decoding (argmax).

    Returns:
      List of generated text strings
    """
    results = []
    if not max_len:
      max_len = self.max_len

    with torch.no_grad():
      for prompt in prompts:
        result = []
        # We only support batch size of 1. This code is just for testing and not meant to be fast/good/etc.
        # Basically, we want training to be -really- easy to play with so we have the minor concession of a kv cache mechanism.
        x, _, _ = self._tokenize_batch([prompt])
        cache = []
        stop = False

        while not stop:
          x, cache = self(x, cache=cache)
          # x shape: (batch, seq_len, vocab_size) -> take last token and squeeze batch
          logits = x[0, -1, :] if x.dim() == 3 else x[-1, :]

          # Apply generation filters in order
          logits = apply_temperature(logits, temperature)
          logits = apply_repetition_penalty(logits, torch.tensor(result, device=logits.device), repetition_penalty)
          logits = top_k_filtering(logits, top_k)
          logits = top_p_filtering(logits, top_p)

          # Sample or argmax
          if do_sample:
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
          else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

          token_id = next_token.squeeze().item()
          result.append(token_id)

          if token_id == self.tokenizer.eot_token or len(result) == max_len:
            stop = True
          else:
            # Prepare input for next iteration: new token as long tensor
            x = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        results.append(self.tokenizer.decode(result))

    return results

  def parameter_count(self) -> int:
    """Count total number of parameters in the model.

    Returns:
      int: Total parameter count
    """
    pc = 0
    for p in self.parameters():
      p_count = 1
      for s in p.shape:
        p_count *= s
      pc += p_count
    return pc

  @classmethod
  def advertise_params(cls) -> dict:
    """Advertise which construction parameters this model type supports.

    This class method returns a dictionary of parameters that can be set via
    config files or command line. Subclasses override this to specify what
    parameters are meaningful for their model.

    Returns:
      dict: Parameter names mapped to descriptions or default values.
            Example: {'num_blocks': 8, 'embedding_dim': 512}
    """
    params = {}
    for param, default in (('max_len', 1024),
                           ('batch_size', 50)):
      if param not in params:
        params[param] = default

      return params


class LMBase(MBase):
  """Base class specifically for language models.
  
  This class extends MBase with a forward signature appropriate for
  autoregressive language modeling with optional key-value caching.
  """
  
  @abstractmethod
  def forward(self, x: torch.Tensor, cache: Optional[List] = None) -> torch.Tensor:
    """Forward pass for language modeling.
    
    Args:
      x: Input token indices of shape (batch_size, sequence_length)
      cache: Optional key-value cache for efficient generation
      
    Returns:
      Output logits and optionally updated cache
    """
    pass


