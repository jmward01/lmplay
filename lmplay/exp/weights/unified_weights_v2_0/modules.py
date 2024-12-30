import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


def gen_mask(max_len: int) -> torch.Tensor:
  return torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)


class ULinear(nn.Module):
  # Modified from pytorch source
  def __init__(self,
               in_features: int,
               out_features: int,
               device=None,
               dtype=None,
               ef=4.0) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    emb_size = int(min(in_features, out_features) * ef)
    mid_size = min(in_features, out_features)
    self.expansion_data = nn.Parameter(torch.empty(emb_size))
    self.bias1 = nn.Linear(emb_size, mid_size)
    self.bias2 = nn.Linear(mid_size, out_features)
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
    bias = self.bias1(self.expansion_data)
    bias = self.bias2(F.gelu(bias))
    result = result + bias
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'
