from torch import nn
import torch
import torch.nn.functional as F


__all__ = ['LRAdd']



class LRAdd(nn.Module):
  def __init__(self, c_dim=None, min_b=None, **kwargs):
    #c_dim looks lik it hurts. NBD.
    super().__init__(**kwargs)
    #Start at 0 so we are balanced
    self.full=not c_dim is None
    self.min_b = min_b
    if not self.full:
      #if min_b is None:
      self.alpha = nn.Parameter(torch.zeros((2,), **kwargs), **kwargs)
      #else:
      #  self.alpha = nn.Parameter(torch.ones((2,), **kwargs), **kwargs)
    else:
      self.alpha = nn.Linear(c_dim * 2, 2, **kwargs)

  def forward(self, x, y):
    if self.full:
      alpha = torch.concat((x,y), -1)
      alpha = self.alpha(alpha)
    else:
      alpha = self.alpha
    #Get alpha between 0 and 2. We want to allow going above 1
    alpha = F.sigmoid(alpha)*2
    if not self.min_b is None:
      #This is less a min and more a point that encourages going up.
      alpha = F.elu(alpha - self.min_b) + self.min_b
      #alpha = F.elu(alpha) + self.min_b
    #else:
    #  alpha = F.sigmoid(alpha)*2
    a = alpha[...,0:1]
    b = alpha[...,1:2]
    return x*a + y*b