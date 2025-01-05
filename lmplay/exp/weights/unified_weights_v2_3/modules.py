import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


def gen_mask(max_len: int) -> torch.Tensor:
  return torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)


class SULinear(nn.Module):
  # Modified from pytorch source
  #This reformulates UW 2.1 to have a shared set of parameters but completely unique sacrificial weights per linear
  def __init__(self,
               expansion_data: nn.Parameter,
               in_features: int,
               out_features: int,
               bias=True,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.has_bias = bias
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    #Hack to avoid storing copies of the expansion data everywhere
    self.expansion_data = [expansion_data]
    self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
    self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    if self.has_bias:
      #Doing the min here because we have things like the vocab that get huge on the out.
      mid_features = int(min(in_features,out_features)**2/expansion_data.size(0))
      self.bias_weights_1 = nn.Linear(expansion_data.size(0), mid_features)
      self.bias_weights_2 = nn.Linear(mid_features, out_features)
      self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    # This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.has_bias:
      bias = F.gelu(self.bias_weights_1(self.expansion_data[0]))
      bias = self.bias_weights_2(bias)
    else:
      bias = None
    result = F.linear(input, self.weight, bias)
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'
