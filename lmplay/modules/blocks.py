"""Transformer encoder block implementations.

This module provides configurable transformer encoder blocks that combine
multi-head attention and feed-forward networks with residual connections.
The blocks are designed to be highly modular, allowing different components
to be swapped out for experimentation.

Key features:
- Configurable attention and feed-forward linear layer types
- Optional layer normalization for attention and MLP components
- Experimental LRAdd (Learned Residual Add) connections
- Support for key-value caching during inference
"""

from torch import nn
from typing import Optional

from .attn import MultiheadAttention
from .general import Add, LRAdd
from lmplay.utils import create_linear

__all__ = ['Block']


class Block(nn.Module):
  """Transformer encoder block with configurable components.
  
  This class implements a standard transformer encoder block consisting of:
  1. Multi-head self-attention with residual connection
  2. Position-wise feed-forward network with residual connection
  3. Layer normalization before each sub-layer (pre-norm architecture)
  
  The block is highly configurable, allowing different linear layer implementations,
  attention mechanisms, and residual connection strategies to be used. This makes
  it ideal for experimenting with architectural variations.
  
  Architecture (with default settings):
  ```
  x -> LayerNorm -> MultiheadAttention -> + -> LayerNorm -> FFN -> + -> output
  |                                       |   |                      |
  +---------------------------------------+   +----------------------+
  ```
  
  The feed-forward network (FFN) follows the standard transformer design:
  Linear(embed_dim, 4*embed_dim) -> GELU -> Linear(4*embed_dim, embed_dim)
  """

  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear,
               # Passing in the class we want for a linear layer since this can be swapped for different exp
               ff_linear=None,
               mha=MultiheadAttention,
               mha_linear=None,
               lradd=False,
               lradd_simple=None,  # only matters is lradd=True
               lradd_predict=None,  # only matters is lradd=True
               lradd_floor=None,  # only matters is lradd=True
               lradd_ceil=None,  # only matters is lradd=True
               ln_attn=True,
               ln_mlp=True,
               **kwargs):
    """Initialize a transformer encoder block.
    
    Args:
        max_len (int): Maximum sequence length, needed for attention mask generation.
        num_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension. Must be divisible by num_heads.
        attn_dropout (float, optional): Dropout probability for attention weights.
            Defaults to 0.1.
        ff_dropout (float, optional): Dropout probability for feed-forward network.
            Defaults to 0.1.
        linear (type): Default linear layer class to use throughout the block.
            Can be overridden for specific components. Defaults to nn.Linear.
        ff_linear (type, optional): Specific linear layer class for feed-forward
            network. If None, uses the general linear parameter.
        mha (type): Multi-head attention class to use. Defaults to MultiheadAttention.
        mha_linear (type, optional): Specific linear layer class for attention
            projections. If None, uses the general linear parameter.
        lradd (bool): If True, use LRAdd (Learned Residual Add) instead of
            standard residual connections. Defaults to False.
        lradd_simple (bool, optional): LRAdd parameter for simple mode.
            Only used if lradd=True.
        lradd_predict (bool, optional): LRAdd parameter for prediction mode.
            Only used if lradd=True.
        lradd_floor (float, optional): LRAdd parameter for minimum weight value.
            Only used if lradd=True.
        lradd_ceil (float, optional): LRAdd parameter for maximum weight value.
            Only used if lradd=True.
        ln_attn (bool): If True, apply layer normalization before attention.
            Defaults to True.
        ln_mlp (bool): If True, apply layer normalization before MLP.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to the attention module.
    """
    super().__init__()
    if ff_linear is None:
      ff_linear = linear
    if mha_linear is None:
      mha_linear = linear
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = nn.Identity()
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = nn.Identity()
    if lradd:
      self.ff_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.ff_lradd = Add()

    if lradd:
      self.mha_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.mha_lradd = Add()

    self.attn = mha(max_len,
                    num_heads,
                    embed_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    linear=mha_linear,
                    **kwargs)
    self.ff = nn.Sequential(create_linear(ff_linear, 'block_ff_1', embed_dim, embed_dim * 4),
                            nn.GELU(),
                            create_linear(ff_linear, 'block_ff_2', embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))

  def forward(self, x, cache: Optional[list] = None):
    """Forward pass through the transformer block.
    
    Applies multi-head self-attention followed by a position-wise feed-forward
    network, with residual connections and layer normalization.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        cache (Optional[list]): Key-value cache for the attention mechanism.
            Used during autoregressive generation to cache previous attention states.
            Should be None during training. If provided, will be modified in place.
            Defaults to None.
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
            after applying attention and feed-forward transformations.
    
    Note:
        The residual connections can be either standard additions or learned
        weighted combinations (LRAdd) based on the block configuration.
    """
    x = self.mha_lradd(x, self.attn(self.ln1(x), cache=cache))
    x = self.ff_lradd(x, self.ff(self.ln2(x)))
    return x
