import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from typing import Optional

class NNEmbedding(nn.Module):
  def __init__(self, cells, num_heads, in_features, embedding_dim, linear=nn.Linear, softmax=True, **kwargs):
    super().__init__()
    self.softmax = softmax
    #Way too much costly logc in here.
    self.head_size = int(embedding_dim/num_heads)
    mid_features = int((in_features + cells)/2)
    self.num_heads = num_heads
    self.cell_count = cells
    self.embedding_dim = embedding_dim
    self.embedding = nn.Parameter(torch.empty((1, self.num_heads, self.cell_count, self.head_size), **kwargs))
    self.selector_1 = linear(in_features, mid_features, **kwargs)
    self.selector_2 = linear(mid_features, cells*num_heads, **kwargs)
    #self.selector = linear(in_features, cells*num_heads, **kwargs)
    self.proj = linear(embedding_dim, embedding_dim)
    init.normal_(self.embedding)

  def forward(self, x):
    batch_size, seq_len, emb_size = x.shape

    s = self.selector_1(x)
    #attn = batch num heads, seq length, value_length
    s = self.selector_2(F.gelu(s)).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    #s = self.selector(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    #embedding = 1 num_heads, cell_count, head_size
    if self.softmax:
      s = F.softmax(s, dim=-1)
    x = torch.matmul(s, self.embedding)

    x = x.transpose(1, 2)
    # y.shape == (batch_size, seq_len, num_heads, head_dim)
    x = x.reshape(batch_size, seq_len, -1)
    x = self.proj(x)
    return x


from lmplay.utils import create_linear


class NNMBlock(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

  """

  def __init__(self,
               num_heads: int,
               cells:int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear,
               # Passing in the class we want for a linear layer since this can be swapped for different exp
               ff_linear=None,
               ln_attn=True,
               ln_mlp=True,
               **kwargs):
    super().__init__()
    if ff_linear is None:
      ff_linear = linear
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x: x
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x: x

    self.attn = NNEmbedding(cells, num_heads, embed_dim, embed_dim, **kwargs)
    self.ff = nn.Sequential(create_linear(ff_linear, 'block_ff_1', embed_dim, embed_dim * 4),
                            nn.GELU(),
                            create_linear(ff_linear, 'block_ff_2', embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))

  def forward(self, x, cache: Optional[list] = None):
    # A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    # the 'cache' is the kv cache and is only needed for inference, not training.
    x = x + self.attn(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x
