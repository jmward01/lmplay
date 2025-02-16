import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class NNEmbedding(nn.Module):
  def __init__(self, cells, embedding_dim, linear=nn.Linear, **kwargs):
    super().__init__()
    self.cell_count = cells
    self.embedding_dim = embedding_dim
    self.embedding = nn.Parameter(torch.empty((1, 1, cells, embedding_dim), **kwargs))
    self.selector = linear(embedding_dim, cells, **kwargs)
    init.normal_(self.embedding)

  def forward(self, x):
    s = self.selector(x).unsqueeze(-1)
    s = torch.softmax(s, dim=-2)
    e = self.embedding * s
    e = torch.sum(e, dim=-2)

    return e


class NNELinear(nn.Module):
  def __init__(self, cells:int, in_features: int, out_features: int, **kwargs):
    super().__init__()
    self.w = nn.Linear(in_features, out_features, **kwargs)
    self.nne = NNEmbedding(cells, out_features)

  def forward(self, x):
    x = self.w(x)
    x = x + self.nne(F.gelu(x))
    return x
