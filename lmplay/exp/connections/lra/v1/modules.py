

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from lmplay.modules import MultiheadAttention


from lmplay.utils import create_linear

class LRAdd(nn.Module):
  def __init__(self, c_dim=None, **kwargs):
    super().__init__(**kwargs)
    #Start at 0 so we are balanced
    self.alpha = nn.Parameter(torch.zeros((2,), **kwargs), **kwargs)
    if c_dim is None:
      self.register_buffer('c', None)
    else:
      self.c = nn.Parameter(torch.zeros(c_dim))

  def forward(self, x, y):
    alpha = F.sigmoid(self.alpha)*2

    if not self.c is None:
      return x*alpha[0] + y*alpha[1] + self.c
    return x*alpha[0] + y*alpha[1]

class Block(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

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
               mha_linear=None,
               ln_attn=True,
               ln_mlp=True,
               add_c=False,
               **kwargs):
    super().__init__()
    if ff_linear is None:
      ff_linear = linear
    if mha_linear is None:
      mha_linear = linear
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x: x
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x: x
    if add_c:
      self.attn_add = LRAdd(c_dim=embed_dim)
      self.ff_add = LRAdd(c_dim=embed_dim)
    else:
      self.attn_add = LRAdd()
      self.ff_add = LRAdd()
    self.attn = MultiheadAttention(max_len,
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
    # A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    # the 'cache' is the kv cache and is only needed for inference, not training.
    x = self.attn_add(x, self.attn(self.ln1(x), cache=cache))
    x = self.ff_add(x, self.ff(self.ln2(x)))
    return x
