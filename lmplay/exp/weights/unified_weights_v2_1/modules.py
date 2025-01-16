import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


def gen_mask(max_len: int) -> torch.Tensor:
  return torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)


class SULinear(nn.Module):
  # Modified from pytorch source
  #This combines UW 1.0 and UW 2.0 so shared sacrificial bias gen and mbias
  def __init__(self,
               shared_mid_weights: nn.Linear,
               in_features: int,
               out_features: int,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    #Hack to avoid storing copies of the mid weights everywhere
    self.shared_mid_weights = [shared_mid_weights]
    self.expansion_data = nn.Parameter(torch.empty(shared_mid_weights.in_features))
    self.bias_weights = nn.Linear(shared_mid_weights.out_features, out_features)
    self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    # This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(self.expansion_data, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    result = F.linear(input, self.weight, None)
    bias = self.shared_mid_weights[0](self.expansion_data)
    bias = self.bias_weights(F.gelu(bias))
    result = result * (self.mbias + self.mbias_bias) + (bias + self.bias_bias)
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'
