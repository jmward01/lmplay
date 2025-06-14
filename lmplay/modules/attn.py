"""Multi-head attention implementations with experimental features.

This module provides flexible multi-head attention implementations that support:
- Swappable linear layer implementations for experimenting with different weight types
- Optional layer normalization on query, key, and value projections
- Learnable scaling factors for attention scores
- Support for both causal and non-causal attention
- Key-value caching for efficient inference
- Cross-attention capabilities

The implementations are designed to be drop-in replacements for standard PyTorch
multi-head attention while providing additional hooks for experimentation.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from lmplay.utils import create_linear, gen_mask

__all__ = ['MultiheadAttention']


class MultiheadAttention(nn.Module):
  """Flexible multi-head attention implementation for experimentation.
  
  This class provides a configurable multi-head attention mechanism that supports
  various experimental features while maintaining compatibility with standard
  transformer architectures. It allows testing different linear layer implementations,
  normalization strategies, and scaling mechanisms.
  
  Key features:
  - Swappable linear layer implementations for Q, K, V, and output projections
  - Optional layer normalization on query, key, and value projections
  - Learnable or fixed scaling of attention scores
  - Support for both causal (autoregressive) and non-causal attention
  - Key-value caching for efficient autoregressive generation
  - Cross-attention support (when x_cross is provided)
  
  The implementation provides both a reference version using PyTorch's
  scaled_dot_product_attention and a custom implementation for better
  visibility into the attention mechanism.
  """

  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               # This was giving me problems. I probably was using it wrong. Maybe I will fix this in the future. For now it is ignored.
               ff_dropout: Optional[float] = 0.1,
               force_ref=False,
               norm_v=False,
               norm_k=False,
               norm_q=False,
               linear=nn.Linear,
               query_linear=None,
               key_linear=None,
               proj_linear=None,
               value_linear=None,
               causal=True,
               learn_scale: bool|str=False):  # Passing in the class we want for a linear layer since this can be swapped for different exp
    """Initialize multi-head attention module.
    
    Args:
        max_len (int): Maximum sequence length for mask generation. This is needed
            for causal attention masks but better implementations could generate
            masks dynamically.
        num_heads (int): Number of attention heads. The embed_dim must be divisible
            by this value.
        embed_dim (int): Embedding dimension. Must be divisible by num_heads to
            determine the dimension per head.
        attn_dropout (float, optional): Dropout probability for attention weights.
            Currently not implemented but kept as placeholder. Defaults to 0.1.
        ff_dropout (float, optional): Dropout probability for the output projection.
            Defaults to 0.1.
        force_ref (bool): If True, forces use of PyTorch's reference implementation
            (scaled_dot_product_attention) when available. Defaults to False.
        norm_v (bool): If True, applies layer normalization to value projections.
            Defaults to False.
        norm_k (bool): If True, applies layer normalization to key projections.
            Defaults to False.
        norm_q (bool): If True, applies layer normalization to query projections.
            Defaults to False.
        linear (type): Linear layer class or factory function to use for creating
            linear projections. Defaults to nn.Linear.
        query_linear (type, optional): Specific linear layer type for query projection.
            If None, uses the general linear parameter.
        key_linear (type, optional): Specific linear layer type for key projection.
            If None, uses the general linear parameter.
        proj_linear (type, optional): Specific linear layer type for output projection.
            If None, uses the general linear parameter.
        value_linear (type, optional): Specific linear layer type for value projection.
            If None, uses the general linear parameter.
        causal (bool): If True, uses causal (autoregressive) attention masking.
            Defaults to True.
        learn_scale (bool|str): Controls learnable scaling of attention scores.
            - False: Use standard 1/sqrt(d_k) scaling
            - True or 'full': Learn per-head, per-position scaling factors
            - 'simple': Learn a single global scaling factor
            Defaults to False.
    """
    # The mask here is generated by the 'casual' value.
    # There are a lot of good reasons to generate a custom mask but for simplicity and clarity this class will deal with mask generation as needed.
    super().__init__()
    if query_linear is None:
      query_linear = linear
    if key_linear is None:
      key_linear = linear
    if value_linear is None:
      value_linear = linear
    if proj_linear is None:
      proj_linear = linear

    assert embed_dim % num_heads == 0, "Embed dim must be a multiple of num_heads."
    self.num_heads = num_heads
    # k&v are what are 'attended' to and will be cached for generation.
    self.key = create_linear(key_linear, 'mha_key', embed_dim, embed_dim)
    self.value = create_linear(value_linear, 'mha_value', embed_dim, embed_dim)
    head_size = int(embed_dim / num_heads)
    if norm_v:
      self.value_norm = nn.LayerNorm(head_size)
    else:
      self.value_norm = lambda x: x

    if norm_k:
      self.key_norm = nn.LayerNorm(head_size)
    else:
      self.key_norm = lambda x: x

    if norm_q:
      self.query_norm = nn.LayerNorm(head_size)
    else:
      self.query_norm = lambda x: x

    self.query = create_linear(query_linear, 'mha_query', embed_dim, embed_dim)

    # proj to clean things up after
    self.proj = create_linear(proj_linear, 'mha_proj', embed_dim, embed_dim)
    if learn_scale == True or learn_scale == 'full':
      self.scale = create_linear(linear, 'mha_scale', embed_dim, num_heads)
      self.scale_type = 'full'
    elif learn_scale == 'simple':
      self.scale = nn.Parameter(torch.tensor([1.0]))
      self.scale_type = 'simple'
    else:
      self.scale_type = None
      self.scale = None
    # I had implementation issues with this dropout so I just... dropped it out.
    # It isn't critical to the concept of MHA so this is the easy route.
    # self.attn_dropout = nn.Dropout(attn_dropout)
    self.proj_dropout = nn.Dropout(ff_dropout)

    # not needed for ref version
    if causal:
      self.register_buffer("mask", gen_mask(max_len))
    else:
      self.register_buffer("mask", None)
    self.force_ref = force_ref

  def _kv_cache_prep(self, cache: Optional[list]) -> bool:
    """Prepare the key-value cache for storing attention states.
    
    This method initializes the cache structure if needed and determines
    whether the cache contains previous key-value pairs that should be
    concatenated with new ones.
    
    Args:
        cache (Optional[list]): A list that will store [keys, values] tensors.
            Modified in place. If None, caching is disabled.
    
    Returns:
        bool: True if the cache contains previous key-value pairs that should
            be concatenated with new computations, False otherwise.
    """
    if cache is None:
      return False
    if len(cache) == 0:
      cache.extend([None, None])
      return False
    return True

  def forward(self, x, x_cross=None, cache: Optional[list] = None) -> torch.Tensor:
    """Perform multi-head attention forward pass.
    
    This method computes multi-head attention with support for both self-attention
    and cross-attention modes. It handles key-value caching for efficient
    autoregressive generation and supports both PyTorch's optimized implementation
    and a custom implementation for experimentation.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            This is used to compute queries in all cases, and also keys/values
            for self-attention.
        x_cross (torch.Tensor, optional): Cross-attention input of shape
            (cross_batch_size, cross_seq_len, embed_dim). When provided, keys and
            values are computed from this tensor instead of x. Defaults to None.
        cache (Optional[list]): Key-value cache for autoregressive generation.
            Should be an empty list on first call. The method will populate it
            with [keys, values] and update it on subsequent calls. Modified in place.
            Not supported for cross-attention. Defaults to None.
    
    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len, embed_dim)
            after applying multi-head attention and output projection.
    
    Note:
        The implementation automatically chooses between causal and non-causal
        attention based on the sequence length and mask settings. For single
        token generation (seq_len=1), causal masking is not applied since
        there are no future tokens to mask.
    """

    if not x_cross is None:
      assert self.mask is None, f"Attn created with a causal mask but passed cross attn inputs"
      # There is a strong argument to support x_attn k,v but I'm not implementing it yet
      assert cache is None, f"Cached k, v aren't supported for x attn"
      x_target = x_cross.unsqueeze(0)
    else:
      x_target = x
    # Useful for later ops
    batch_size, seq_len, embed_dim = x.shape
    target_batch_size, target_seq_len, target_embed_dim = x_target.shape
    if self.scale_type is None and (self.force_ref or torch.cuda.is_available()):
      k = self.key_norm(self.key(x_target).reshape(target_batch_size, target_seq_len, self.num_heads, -1)).transpose(1,
                                                                                                                     2)
      v = self.value_norm(
        self.value(x_target).reshape(target_batch_size, target_seq_len, self.num_heads, -1)).transpose(1, 2)
      q = self.query_norm(self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
      # Not adding dropout to match the cpu version
      # y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout, is_causal=self.casual)
      if self._kv_cache_prep(cache):
        k = torch.concat((cache[0], k), dim=2)
        v = torch.concat((cache[1], v), dim=2)
      if cache:
        # Update the cache with the new k&v from this token generation.
        cache[0] = k
        cache[1] = v

      if seq_len > 1 and not self.mask is None:
        # We are either generating the prompt or are in training with causal attn so we need the casual mask
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
      else:
        # No need to mask since we are just generating the next token.
        # This is broken for batch size > 1 but that is fine for the simple test inference that this project implements.
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
      # This appears to be a little faster on non-cuda hardware for training than the pytorch ref implementation.
      # This also opens up how mha works to make it easier to play with parts of it.

      k = self.key_norm(self.key(x_target).reshape(target_batch_size, target_seq_len, self.num_heads, -1)).permute(0, 2,
                                                                                                                   3, 1)
      v = self.value_norm(
        self.value(x_target).reshape(target_batch_size, target_seq_len, self.num_heads, -1)).transpose(1, 2)
      q = self.query_norm(self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)

      if self._kv_cache_prep(cache):
        k = torch.concat((cache[0], k), dim=-1)
        v = torch.concat((cache[1], v), dim=2)
      if cache:
        # Update the cache with the new k&v from this token generation.
        cache[0] = k
        cache[1] = v

      # This is where the 'scaled' is implemented!
      if self.scale_type is None:
        scale =  math.sqrt(q.size(-1))
      elif self.scale_type == 'full':

        #We want the scale on the same dimension as the softmax. That means the last value is the scale
        #scale should become: batch, heads, seq length, scale
        #Since we softmax of -1 we are scaling on -1
        scale = F.sigmoid(self.scale(x)).transpose(1,2).unsqueeze(-1) * (2 * math.sqrt(q.size(-1)))
      elif self.scale_type == 'simple':
        scale = math.sqrt(q.size(-1)) * self.scale
      else:
        raise  ValueError(f"Unknown scale_type {self.scale_type}")
      attn = torch.matmul(q, k) / scale
      # No need to mask for seq len 1 since we are just generating the next token.
      if seq_len > 1 and not self.mask is None:
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))

      # This dropout caused issues on CPU. Didn't spend a lot of time debugging and it isn't critical to the logic of mha.
      # If you are building something that will really be prod ready you should consider forcing ref and adding attn_droupout back
      # or better yet just use a full ref implementation of MHA instead of this class.
      ##attn = self.attn_dropout(attn)
      # attn.shape == (batch_size, num_heads, seq_len, seq_len)
      attn = F.softmax(attn, dim=-1)
      x = torch.matmul(attn, v)

    x = x.transpose(1, 2)
    # y.shape == (batch_size, seq_len, num_heads, head_dim)
    x = x.reshape(batch_size, seq_len, -1)
    # y.shape == (batch_size, seq_len, embed_dim)
    x = self.proj_dropout(self.proj(x))
    return x
