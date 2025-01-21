import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

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
    if self.bias is not None:
      bias = self.bias + self.bias_bias
    else:
      bias = None
    weight = self.weight.t() * (self.mbias + self.mbias_bias)
    #This can clearly be re-stored as a normal weight/bias for prod.
    result = F.linear(input, weight.t(), bias)
    return result


  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'

class DULinear(nn.Module):
  #Deep Unified Linear
  def __init__(self,
               in_features: int,
               out_features: int,
               bias = True,
               device=None,
               dtype=None,
               bias_exp_mul=8.0,
               bias_mid_mul=1.0,
               mbias_exp_mul=8,
               mbias_mid_mul=1.0,
               linear=nn.Linear) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.has_bias = bias
    self.in_features = in_features
    self.out_features = out_features
    if mbias_exp_mul > 0:
      #They want to predict the mbias
      mbias_expansion_features = int(in_features*mbias_exp_mul)
      mbias_weights_hidden = int(min(in_features, out_features) * mbias_mid_mul)
      self.mbias_expansion_data = nn.Parameter(torch.empty(mbias_expansion_features))
      self.mbias_weights_1 = linear(mbias_expansion_features, mbias_weights_hidden)
      self.mbias_weights_2 = linear(self.mbias_weights_hidden, out_features)
      self.register_parameter("mbias", None)
    else:
      #no mbias prediction, just make parameters
      self.register_parameter("mbias_expansion_data", None)
      self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    #we always have an mbias_bais
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    #Hey, look! Normal weights!
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    if self.has_bias == True:
      #So they want a bias eh? Do they want to predict it or just use a static bias?
      if bias_exp_mul > 0:
        bias_expansion_features = int(in_features*bias_exp_mul)
        bias_weights_hidden = int(min(in_features, out_features) * bias_mid_mul)
        self.bias_expansion_data = nn.Parameter(torch.empty(bias_expansion_features))
        self.bias_weights_1 = linear(bias_expansion_features, bias_weights_hidden)
        self.bias_weights_2 = linear(self.bias_weights_hidden, out_features)
      else:
        self.register_parameter("bias_expansion_data", None)
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
      self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    #This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    if self.bias is not None:
      init.uniform_(self.bias, -bound, bound)
    if self.bias_expansion_data is not None:
      init.uniform_(self.bias_expansion_data, -bound, bound)
    if self.mbias_expansion_data is not None:
      init.uniform_(self.mbias_expansion_data, -bound, bound)


  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.bias is not None:
      bias = self.bias + self.bias_bias
    elif self.bias_expansion_data is not None:
      bias = F.gelu(self.bias_weights_1(self.bias_expansion_data))
      bias = self.bias_weights_2(bias)
    else:
      bias = None

    if self.mbias is not None:
      mbias = self.mbias
    else:
      mbias = F.gelu(self.mbias_weights_1(self.mbias_expansion_data))
      mbias = self.mbias_weights_2(mbias)

    weight = self.weight.t() * (mbias + self.mbias_bias)
    result = F.linear(input, weight.t(), bias)
    return result


  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'