"""General utility modules for neural network experimentation.

This module provides utility components that can be used throughout the
architecture for various experimental purposes:

- LRAdd: Learned residual connections with adaptive weighting
- CAdd: Configurable addition operations (standard or learned)
- NopModule: No-operation module for architectural flexibility
- hide_net: Utility to hide networks from PyTorch serialization
"""

from torch import nn
import torch
import torch.nn.functional as F
from lmplay.utils import ignore_default, DEFAULT

__all__ = ['LRAdd', 'hide_net', 'NopModule']



class LRAdd(nn.Module):
  """Learned Residual Addition with adaptive weighting.
  
  LRAdd (Learned Residual Add) replaces standard residual connections (x + y)
  with learned weighted combinations (alpha*x + beta*y). The weights can be
  either fixed parameters or predicted based on the input values.
  
  This allows the model to learn optimal blending strategies for residual
  connections, potentially improving gradient flow and feature combination.
  
  Modes:
  - Simple: Uses 2 global weights (alpha, beta) for all positions
  - Full: Uses 2*features weights for position-specific blending
  - Predicted: Weights are functions of the concatenated inputs
  """
  
  @ignore_default
  def __init__(self,
               features:int,
               floor=0.4,
               ceil=1.5,
               simple=True,
               predict=True,
               **kwargs):
    """Initialize LRAdd module.
    
    Args:
        features (int): Number of features in the input tensors.
        floor (float): Minimum value for sigmoid-scaled weights. Defaults to 0.4.
        ceil (float): Maximum value for sigmoid-scaled weights. Defaults to 1.5.
        simple (bool): If True, use global weights. If False, use per-feature
            weights. Defaults to True.
        predict (bool|str): How to determine weights:
            - True: Predict weights with linear layer from concatenated inputs
            - "mlp": Predict weights with 2-layer MLP from concatenated inputs  
            - False: Use learned parameter weights
            Defaults to True.
        **kwargs: Additional arguments passed to parent class.
    """
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
    """Compute weighted combination of two tensors.
    
    Args:
        x (torch.Tensor): First input tensor (typically the residual path).
        y (torch.Tensor): Second input tensor (typically the transformed path).
    
    Returns:
        torch.Tensor: Weighted combination alpha*x + beta*y where alpha and beta
            are learned or predicted weights scaled to [floor, ceil].
    """
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
  """Configurable addition operation for residual connections.
  
  This module provides a unified interface for different types of
  residual connections, making it easy to experiment with various
  addition strategies.
  """
  
  @ignore_default
  def __init__(self, features:int, add_type:str|None = "A"):
    """Initialize configurable addition.
    
    Args:
        features (int): Number of features (used for LRAdd mode).
        add_type (str|None): Type of addition operation:
            - None: Pass through y only (no residual)
            - "A": Standard addition (x + y)
            - "L": Learned residual addition using LRAdd
            Defaults to "A".
    """
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
    """Apply the configured addition operation.
    
    Args:
        x (torch.Tensor): First tensor (residual path).
        y (torch.Tensor): Second tensor (main path).
    
    Returns:
        torch.Tensor: Result of the configured operation.
    """
    return self.add_f(x, y)


class NopModule(nn.Module):
  """No-operation module that returns None.
  
  This module is useful for architectural flexibility where a module
  slot needs to be filled but shouldn't perform any computation.
  It allows for cleaner conditional architecture definitions.
  """
  
  def forward(self, *args, **kwargs):
    """Return None regardless of inputs.
    
    Args:
        *args: Any positional arguments (ignored).
        **kwargs: Any keyword arguments (ignored).
    
    Returns:
        None: Always returns None.
    """
    return None


def hide_net(net: nn.Module):
  """Hide a network module from PyTorch's serialization mechanism.
  
  This utility function wraps a module in a lambda function to prevent
  PyTorch from including it in state_dict serialization. This is useful
  for shared networks that should not be saved multiple times or for
  temporary networks used only during training.
  
  Args:
      net (nn.Module): The network module to hide from serialization.
  
  Returns:
      callable: A lambda function that forwards calls to the hidden network.
  
  Example:
      shared_net = nn.Linear(100, 50)
      self.hidden_net = hide_net(shared_net)  # Won't be in state_dict
  """
  # just need to hold onto something that PyTorch won't try to serialize/deserialize
  return lambda *args, **kwargs: net(*args, **kwargs)