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
               bias = True,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.has_bias = bias
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    if self.has_bias == True:
      self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
      self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      #this is needed because?????? Won't work in some frameworks without it because they are constructing the models and not the model code.
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    #This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      init.uniform_(self.bias, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    #calculate don't add the bias yet
    sum_mbias = self.mbias + self.mbias_bias
    if self.bias is not None:
      sum_bias = self.bias + self.bias_bias
    else:
      sum_bias = None
    result = F.linear(input, self.weight, None)
    #We apply this mbias idea to both the weights and the biases. For the weights though this works out as a mul and not an add.
    if self.has_bias:
      result = result * sum_mbias + sum_bias
    else:
      result = result * sum_mbias

    #Is this mathematically equivelant to mx + b ? yes.
    # We could just do:
    # w = self.weight.t() * sum_mbias
    # w = w.t()
    # result = F.linear(input, w, sum_bias)
    # So obviously the sum_mbias could be saved as a standard bias and the w could be saved as the standard weights and the network would be equivalent and just absorb those extra parameters.
    #If we can have a single parameter that captures the 'bias' of the weights then changes to the weights can be centered on this bias and the gradients don't need to be as big.
    return result


  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'

