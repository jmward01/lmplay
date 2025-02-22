from torch import nn
from typing import Optional
from lmplay.modules import MultiheadAttention


class Block(nn.Module):
  """Just norms the k and v. Shows improvement over no norm, but just v is much better.

  """

  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.attn = MultiheadAttention(max_len,
                                   num_heads,
                                   embed_dim,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   norm_k=True,
                                   norm_v=True)
    self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            nn.Linear(embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))

  def forward(self, x, cache: Optional[list] = None):
    # A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    # the 'cache' is the kv cache and is only needed for inference, not training.
    x = x + self.attn(self.ln1(x), cache=cache)
    x = x + self.ff(self.ln2(x))
    return x
