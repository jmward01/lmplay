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
               predict_mbias2 = False,
               predict_mbias = True,
               predict_bias = True,
               exp_mul=16.0,
               mid_mul=1.0,
               linear=nn.Linear) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    expansion_features = int(in_features*exp_mul)
    weights_hidden = int(min(in_features, out_features) * mid_mul)

    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    if predict_mbias or predict_mbias2 or (predict_bias and bias):
      self.expansion_data = nn.Parameter(torch.empty(expansion_features))
      self.expansion_weights = linear(expansion_features, weights_hidden)
    else:
      self.register_parameter('expansion_data', None)
      self.register_parameter('expansion_weights', None)

    if predict_mbias == True:
      self.mbias_weights = nn.Linear(weights_hidden, in_features)
      self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
      self.register_parameter('mbias', None)
    elif predict_mbias == False:
      self.mbias = nn.Parameter(torch.ones(in_features, **factory_kwargs))
      self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
      self.register_parameter('mbias_weights', None)
    else:
      self.register_parameter('mbias', None)
      self.register_parameter('mbias_bias', None)
      self.register_parameter('mbias_weights', None)

    if predict_mbias2 == True:
      self.mbias2_weights = nn.Linear(weights_hidden, out_features)
      self.mbias2_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
      self.register_parameter('mbias2', None)
    elif predict_mbias2 == False:
      self.mbias2 = nn.Parameter(torch.ones(out_features, **factory_kwargs))
      self.mbias2_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
      self.register_parameter('mbias2_weights', None)
    else:
      self.register_parameter('mbias2', None)
      self.register_parameter('mbias2_bias', None)
      self.register_parameter('mbias2_weights', None)

    if bias == True:
      if predict_bias == True:
        self.bias_weights = nn.Linear(weights_hidden, out_features)
        self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
        self.register_parameter('bias', None)
      elif predict_bias == False:
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
        self.register_parameter('bias_weights', None)
      else:
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.register_parameter('bias_bias', None)
        self.register_parameter('bias_weights', None)
    else:
      self.register_parameter('bias_bias', None)
      self.register_parameter('bias_weights', None)
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
    if self.expansion_data is not None:
      init.uniform_(self.expansion_data, -bound, bound)


  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.expansion_data is not None:
      mid = F.gelu(self.expansion_weights(self.expansion_data))
    else:
      mid = None

    if not self.bias_weights is None:
      bias = self.bias_weights(mid) + self.bias_bias
    elif not self.bias_bias is None:
      bias = self.bias + self.bias_bias
    elif not self.bias is None:
      bias = self.bias
    else:
      bias = None

    if not self.mbias_weights is None:
      mbias = self.mbias_weights(mid) + self.mbias_bias
    elif not self.mbias is None:
      mbias = self.mbias + self.mbias_bias
    else:
      mbias = None

    if not mbias is None:
      weight = self.weight * mbias
    else:
      weight = self.weight

    if not self.mbias2_weights is None:
      mbias2 = self.mbias2_weights(mid) + self.mbias2_bias
    elif not self.mbias2 is None:
      mbias2 = self.mbias2 + self.mbias2_bias
    else:
      mbias2 = None

    if not mbias2 is None:
      weight = weight.t() * mbias2
      result = F.linear(input, weight.t(), bias)
    else:
      result = F.linear(input, weight, bias)
    return result


  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'