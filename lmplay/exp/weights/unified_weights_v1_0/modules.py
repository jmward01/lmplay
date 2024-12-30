import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import init

def gen_mask(max_len:int) -> torch.Tensor:
    return  torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)

class ULinear(nn.Module):
  #Modified from pytorch source
  def __init__(self,
               in_features: int,
               out_features: int,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    #This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(self.bias, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    #don't add the bias yet
    result = F.linear(input, self.weight, None)
    #Is this mathematically equivelant to mx + b ? yes. That isn't the point though. Think of it like this.
    #If we can have a single parameter that captures the 'bias' of the weights then changes to the weights can be centered on this bias and the gradients don't need to be as big.
    #We apply this mbias idea to both the weights and the biases. For the weights though this works out as a mul and not an add.
    result = result * (self.mbias + self.mbias_bias) + (self.bias + self.bias_bias)
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'

