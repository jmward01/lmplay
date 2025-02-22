from torch import nn
import torch
import torch.nn.functional as F


__all__ = ['LRAdd']



class LRAdd(nn.Module):
  def __init__(self, c_dim=None, **kwargs):
    #c_dim looks lik it hurts. NBD.
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