import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class NNEmbedding(nn.Module):
  def __init__(self, cells, num_heads, in_features, embedding_dim, linear=nn.Linear, **kwargs):
    super().__init__()
    self.head_size = int(embedding_dim/num_heads)
    #mid_features = int((in_features + cells)/2)
    self.num_heads = num_heads
    self.cell_count = cells
    self.embedding_dim = embedding_dim
    self.embedding = nn.Parameter(torch.empty((1, self.num_heads, self.cell_count, self.head_size), **kwargs))
    #self.selector_1 = linear(in_features, mid_features, **kwargs)
    #self.selector_2 = linear(mid_features, cells*num_heads, **kwargs)
    self.selector = linear(in_features, cells*num_heads, **kwargs)
    self.proj = linear(embedding_dim, embedding_dim)
    init.normal_(self.embedding)

  def forward(self, x):
    batch_size, seq_len, emb_size = x.shape

    #s = self.selector_1(x)
    #attn = batch num heads, seq length, value_length
    #s = self.selector_2(F.gelu(s)).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    s = self.selector(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    #embedding = 1 num_heads, cell_count, head_size
    s = F.softmax(s, dim=-1)
    x = torch.matmul(s, self.embedding)

    x = x.transpose(1, 2)
    # y.shape == (batch_size, seq_len, num_heads, head_dim)
    x = x.reshape(batch_size, seq_len, -1)
    x = self.proj(x)
    return x


class NNELinear(nn.Module):
  def __init__(self, cells:int, num_heads, in_features: int, out_features: int,  **kwargs):
    super().__init__()
    self.w = nn.Linear(in_features, out_features, **kwargs)
    self.nne = NNEmbedding(cells, num_heads, in_features, in_features)

  def forward(self, x):
    #x = self.w(x)
    #x = x + self.nne(x)
    x = self.w(x) + self.nne(x)
    return x
