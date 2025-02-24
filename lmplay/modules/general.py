from torch import nn
import torch
import torch.nn.functional as F


__all__ = ['LRAdd']



class LRAdd(nn.Module):
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
    batch, sequence, features = weights.shape
    weights = weights.reshape(batch, sequence, 2, -1)
    alpha = weights[:,:,0,:]
    beta = weights[:,:,1,:]

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