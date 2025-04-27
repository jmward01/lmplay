from torch import nn
import torch
import torch.nn.functional as F
from lmplay.utils import ignore_default, DEFAULT

__all__ = ['LRAdd', 'hide_net', 'NopModule']



class LRAdd(nn.Module):
  @ignore_default
  def __init__(self,
               features:int,
               floor=0.4,
               ceil=1.5,
               simple=True,
               predict=True,
               **kwargs):
    #c_dim looks lik it hurts. NBD.
    super().__init__(**kwargs)
    if floor is None:
      floor = 0.4
    if ceil is None:
      ceil = 1.5
    if simple is None:
      simple = True
    if predict is None:
      predict = True

    self.features = features
    self.floor=floor
    self.ceil=ceil
    self.predict = predict
    self.simple = simple
    if self.simple:
      self.out_features = 2
    else:
      self.out_features = 2*features

    if self.predict == True:
      self.weights = nn.Linear(features * 2, self.out_features, **kwargs)
    elif self.predict == "mlp":
      mid_features = features
      self.weights = nn.Sequential(nn.Linear(features * 2, mid_features, **kwargs),
                                   nn.ReLU(),
                                   nn.Linear(mid_features, self.out_features, **kwargs))
    elif self.predict == False:
      self.weights = nn.Parameter(torch.zeros((self.out_features,), **kwargs), **kwargs)
    else:
      raise ValueError(f"Unknown predict type {self.predict}")


  def forward(self, x, y):
    if self.predict:
      weights = torch.concat((x,y), -1)
      weights = self.weights(weights)
    else:
      weights = self.weights
      #no batch or sequence
      weights = weights.reshape(1, 1, -1)
    #at this point alpha is: batch, seq, weights.
    #Let's apply the floor/ceil/target logic
    scale = self.ceil - self.floor
    weights = F.sigmoid(weights)*scale + self.floor
    shape = weights.shape[:-1]
    weights = weights.reshape(*shape, 2, -1)
    if len(shape) == 2:
      alpha = weights[:,:,0,:]
      beta = weights[:,:,1,:]
    else:
      alpha = weights[:,0,:]
      beta = weights[:,1,:]

    ##Now we split between the alpha and beta
    ##Get alpha between 0 and 2. We want to allow going above 1
    #weights = F.sigmoid(weights)*2
    #if not self.min_b is None:
    #  #This is less a min and more a point that encourages going up.
    #  weights = F.elu(weights - self.min_b) + self.min_b
    #  #alpha = F.elu(alpha) + self.min_b
    ##else:
    ##  alpha = F.sigmoid(alpha)*2
    #a = weights[...,0:1]
    #b = weights[...,1:2]
    return x*alpha + y*beta


class CAdd(nn.Module):
  @ignore_default
  def __init__(self, features:int, add_type:str|None = "A"):
    super().__init__()
    if add_type is None:
      self.add_f = lambda x, y:y
    elif add_type == "A":
      self.add_f = lambda x, y: x + y
    elif add_type == "L":
      self.add_f = LRAdd(features)
    else:
      raise ValueError(f"Unknown add {add_type}")

  def forward(self, x, y):
    return self.add_f(x, y)


class NopModule(nn.Module):
  def forward(self, *args, **kwargs):
    return None


def hide_net(net: nn.Module):
  # just need to hold onto something that PyTorch won't try to serialize/deserialize
  return lambda *args, **kwargs: net(*args, **kwargs)