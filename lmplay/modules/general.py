from torch import nn
import torch
import torch.nn.functional as F


__all__ = ['LRAdd']



class LRAdd(nn.Module):
  def __init__(self, c_dim=None, **kwargs):
    #c_dim looks lik it hurts. NBD.
    super().__init__(**kwargs)
    #Start at 0 so we are balanced
    self.full=not c_dim is None
    if not self.full:
      self.alpha = nn.Parameter(torch.zeros((2,), **kwargs), **kwargs)
    else:
      self.alpha = nn.Linear(c_dim * 2, 2, **kwargs)

  def forward(self, x, y):


    if self.full:
      alpha = torch.concat((x,y), -1)
      alpha = F.sigmoid(self.alpha(alpha))*2
      r =  x*alpha[:,:,0:1] + y*alpha[:,:,1:2]
      return r
    alpha = self.alpha
    alpha = F.sigmoid(alpha)*2

    return x*alpha[0] + y*alpha[1]