from torch import nn
from typing import Optional
from lmplay.utils import create_linear
from lmplay.modules import MultiheadAttention, UnifiedEmbedding


class CachedOutput(nn.Module):
  def __init__(self, module:nn.Module):
    super().__init__()
    self.module = [module]
    self.cached_value = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_value = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def forward(self, *args, **kwargs):
    if self.cached_value is None:
      self.cached_value = self.module[0]()
    return self.cached_value

class Block(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

  """
  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear, #Passing in the class we want for a linear layer since this can be swapped for different exp
               ln_attn=True,
               ln_mlp=True,
               nnm:UnifiedEmbedding|int=32,
               front_emb_mul=64,
               x_cross_ff=True):
    super().__init__()
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
      self.ln1_cross = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x:x
      self.ln1_cross = lambda x:x
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x:x

    if x_cross_ff:
      self.ln2_cross = nn.LayerNorm(embed_dim)
      self.ff_cross = nn.Sequential(create_linear(linear, 'block_ff_1', embed_dim, embed_dim * 4),
                              nn.GELU(),
                              create_linear(linear, 'block_ff_2', embed_dim * 4, embed_dim),
                              nn.Dropout(ff_dropout))

    else:
      self.ln2_cross = lambda x:x
      self.register_module('ff_cross', None)
    if isinstance(nnm, int):
      self.nnm = UnifiedEmbedding(nnm, embed_dim, front_emb_mul)
      nnm = self.nnm
    else:
      self.register_module('nnm', None)
    #This is what we will use.
    # If they passed in a ashared embedding then we don't want to set it directly since then it will become part of our state dict.
    self._nnm = CachedOutput(nnm)

    self.x_attn = MultiheadAttention(max_len,
                                   num_heads,
                                   embed_dim,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   linear=linear,
                                   causal=False)

    self.attn = MultiheadAttention(max_len,
                                   num_heads,
                                   embed_dim,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   linear=linear)


    self.ff = nn.Sequential(create_linear(linear, 'block_ff_1', embed_dim, embed_dim * 4),
                            nn.GELU(),
                            create_linear(linear, 'block_ff_2', embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))



  def forward(self, x, cache:Optional[list]=None):
    #A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    #the 'cache' is the kv cache and is only needed for inference, not training.
    x = x + self.x_attn(self.ln1_cross(x), x_cross=self._nnm())
    if not self.ff_cross is None:
      x = x + self.ff_cross(self.ln2_cross(x))
    x = x + self.attn(self.ln1(x), cache=cache)
    x = x + self.ff(self.ln2(x))
    return x
