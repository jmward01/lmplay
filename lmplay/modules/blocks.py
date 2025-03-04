from torch import nn
from typing import Optional

from .attn import MultiheadAttention
from .general import LRAdd
from lmplay.utils import create_linear


__all__ = ['Block']
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
               lradd=False,
               lradd_simple=None, #only matters is lradd=True
               lradd_predict=None, #only matters is lradd=True
               lradd_floor=None, #only matters is lradd=True
               lradd_ceil=None, #only matters is lradd=True
               ln_attn=True,
               ln_mlp=True,
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
    if lradd:
      self.ff_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.ff_lradd = lambda x,y: x + y

    if lradd:
      self.mha_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.mha_lradd = lambda x,y: x + y

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
    x = self.mha_lradd(x, self.attn(self.ln1(x), cache=cache))
    x = self.ff_lradd(x, self.ff(self.ln2(x)))
    return x
