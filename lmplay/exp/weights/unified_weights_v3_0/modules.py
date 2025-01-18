import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import init

def gen_mask(max_len:int) -> torch.Tensor:
    return  torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)

class DULinear(nn.Module):
  #Deep Unified Linear
  def __init__(self,
               in_features: int,
               out_features: int,
               bias = True,
               device=None,
               dtype=None,
               exp_mul=8,
               mid_mul=1) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.has_bias = bias
    self.in_features = in_features
    self.out_features = out_features
    self.expansion_features = int(in_features*exp_mul)
    self.bias_weights_hidden = int(min(in_features, out_features)*mid_mul)


    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    if self.has_bias == True:
      self.expansion_data = nn.Parameter(torch.empty(self.expansion_features))
      self.bias_weights_1 = nn.Linear(self.expansion_features, self.bias_weights_hidden)
      self.bias_weights_2 = nn.Linear(self.bias_weights_hidden, out_features)
      self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      #this is needed because?????? Won't work in some frameworks without it because they are constructing the models and not the model code.
      self.register_parameter("expansion_data", None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    #This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.expansion_data is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      init.uniform_(self.expansion_data, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.expansion_data is not None:
      bias = F.gelu(self.bias_weights_1(self.expansion_data))
      bias = self.bias_weights_2(bias)
      bias = bias + self.bias_bias
    else:
      bias = None
    weight = self.weight.t() * (self.mbias + self.mbias_bias)
    result = F.linear(input, weight.t(), bias)
    return result


  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'

