import math

import torch
from torch import nn
from torch.nn import init, functional as F

from lmplay.modules.general import NopModule, hide_net
from lmplay.utils import set_accepts_purpose

DEFAULT_BOUND = 0.01
DEFAULT_CACHEABLE = False

__all__ = ["ULinear", "DULinear", "SimpleMLP", "MultiMLP", "SPredictor", "SDULinear", "SULinear"]

class ULinear(nn.Module):
  # Modified from pytorch source
  def __init__(self,
               in_features: int,
               out_features: int,
               bias=True,
               bias_bias=True,
               mbias=True,
               imbias=True,
               iambias=True,
               ambias=True,
               device=None,
               dtype=None,
               cacheable=True) -> None:
    self.cacheable = cacheable
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.has_bias = bias
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    if mbias == True:
      self.mbias = nn.Parameter(torch.ones(out_features, **factory_kwargs))
      self.mbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      self.register_parameter('mbias', None)
    if self.has_bias == True:
      self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
      if bias_bias == True:
        self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
      else:
        self.register_parameter("bias_bias", None)
    else:
      # this is needed because?????? Won't work in some frameworks without it because they are constructing the models and not the model code?
      self.register_parameter("bias", None)
      self.register_parameter("bias_bias", None)
    if imbias == True:
      self.imbias = nn.Parameter(torch.ones(in_features, **factory_kwargs))
      self.imbias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      self.register_parameter("imbias", None)
    if ambias == True:
      self.ambias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
      self.ambias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      self.register_parameter("ambias", None)
    if iambias == True:
      self.iambias = nn.Parameter(torch.zeros(in_features, **factory_kwargs))
      self.iambias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    else:
      self.register_parameter("iambias", None)
    self.reset_parameters()
    self.cached_weights = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_weights = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def reset_parameters(self) -> None:
    # This uses the pytorch init, different inits may be valuable with the mbias in place.
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      init.uniform_(self.bias, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.cached_weights is None:
      if self.bias is not None:
        bias = self.bias
        if not self.bias_bias is None:
           bias = bias + self.bias_bias
      else:
        bias = None
      weight = self.weight.t()
      if not self.mbias is None:
        #This should be functionally equivalent to result * (self.mbias + self.mbias_bias)
        #Depending on how you train this will be slower or faster than the alternative.
        # Also, this makes it clear that these weights can be frozen as regular weights and all the sacrificial weights dropped.
        weight = weight * (self.mbias + self.mbias_bias)

      if not self.ambias is None:
        weight = weight + (self.ambias + self.ambias_bias)

      weight = weight.t()
      if not self.imbias is None:
        #This should be functionally equivalent to input * (self.imbias + self.imbias_bias)
        #Depending on how you train this will be slower or faster than the alternative.
        # Also, this makes it clear that these weights can be frozen as regular weights and all the sacrificial weights dropped.
        weight = weight * (self.imbias + self.imbias_bias)

      if not self.iambias is None:
        weight = weight + (self.iambias + self.iambias_bias)

      if self.cacheable:
        self.cached_weights = (weight, bias)
    else:
      weight, bias = self.cached_weights
    # This can clearly be re-stored as a normal weight/bias for prod.
    result = F.linear(input, weight, bias)

    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'


class DULinear(nn.Module):
  # Deep Unified Linear
  def __init__(self,
               in_features: int,
               out_features: int,
               bias=True,
               device=None,
               dtype=None,
               predict_bias=True,
               predict_mbias=True,
               predict_mbias2=False,
               predict_mbias_a=None,
               predict_mbias2_a=None,
               exp_mul=16.0,
               mid_mul=1.0,
               expansion_weights=True,
               linear=nn.Linear,
               cacheable=True) -> None:
    self.cacheable = cacheable
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    expansion_features = int(in_features * exp_mul)
    if expansion_weights:
      weights_hidden = int(min(in_features, out_features) * mid_mul)
    else:
      weights_hidden = expansion_features

    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    if predict_mbias or predict_mbias2 or (predict_bias and bias):
      self.expansion_data = nn.Parameter(torch.empty(expansion_features))
      if expansion_weights:
        self.expansion_weights = linear(expansion_features, weights_hidden)
      else:
        self.register_parameter('expansion_weights', None)
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

    if predict_mbias_a == True:
      self.mbias_a_weights = nn.Linear(weights_hidden, in_features)
      self.register_parameter('mbias_a', None)
    elif predict_mbias_a == False:
      self.mbias_a = nn.Parameter(torch.zeros(in_features, **factory_kwargs))
      self.register_parameter('mbias_a_weights', None)
    else:
      self.register_parameter('mbias_a', None)
      self.register_parameter('mbias_a_weights', None)

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

    if predict_mbias2_a == True:
      self.mbias2_a_weights = nn.Linear(weights_hidden, out_features)
      self.register_parameter('mbias2_a', None)
    elif predict_mbias2_a == False:
      self.mbias2_a = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
      self.register_parameter('mbias2_a_weights', None)
    else:
      self.register_parameter('mbias2_a', None)
      self.register_parameter('mbias2_a_weights', None)

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

    self.cached_weights = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_weights = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def reset_parameters(self) -> None:
    # This uses the pytorch init, different inits may be valuable with the mbias in place.
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
    if self.cached_weights is None:
      if self.expansion_data is not None:
        mid = F.gelu(self.expansion_weights(self.expansion_data))
      elif self.expansion_weights is not None:
        mid = self.expansion_data
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

      if not self.mbias_a_weights is None:
        mbias_a = self.mbias_a_weights(mid)
      elif not self.mbias_a is None:
        mbias_a = self.mbias_a
      else:
        mbias_a = None

      if not mbias is None:
        weight = self.weight * mbias
      else:
        weight = self.weight

      if not mbias_a is None:
        weight = weight + mbias_a

      if not self.mbias2_weights is None:
        mbias2 = self.mbias2_weights(mid) + self.mbias2_bias
      elif not self.mbias2 is None:
        mbias2 = self.mbias2 + self.mbias2_bias
      else:
        mbias2 = None

      if not self.mbias2_a_weights is None:
        mbias2_a = self.mbias2_a_weights(mid)
      elif not self.mbias2_a is None:
        mbias2_a = self.mbias2_a
      else:
        mbias2_a = None

      if not mbias2 is None:
        weight = weight.t() * mbias2
        weight = weight.t()

      if not mbias2_a is None:
        weight = weight.t() + mbias2_a
        weight = weight.t()
      if self.cacheable:
        self.cached_weights = (weight, bias)
    else:
      weight, bias = self.cached_weights
    result = F.linear(input, weight, bias)
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'


class SimpleMLP(nn.Module):
  def __init__(self,
               in_features: int,
               out_features: int,
               bias=True,
               layers=1,
               non_linearity=nn.GELU,
               linear=nn.Linear,
               mid_features=None,
               device=None,
               dtype=None):
    super().__init__()
    assert layers >= 1

    factory_kwargs = {'device': device, 'dtype': dtype}

    self.nonlinearity = non_linearity
    self.in_features = in_features
    self.out_features = out_features
    self.has_bias = bias

    if mid_features is None:
      mid_features = min(in_features, out_features)
    if layers > 1:
      l = [linear(in_features, mid_features, **factory_kwargs), non_linearity()]
      for _ in range(layers - 2):
        l.extend([linear(mid_features, mid_features, **factory_kwargs), non_linearity()])
      l.append(linear(mid_features, out_features, bias=bias, **factory_kwargs))
    else:
      l = [linear(in_features, out_features, bias=bias, **factory_kwargs)]

    self.net = nn.Sequential(*l)

  def forward(self, *arg):
    return self.net(*arg)


class MultiMLP(nn.Module):
  def __init__(self,
               in_features: int,
               mid_features: int,
               layers=1,
               non_linearity=nn.GELU,
               linear=nn.Linear,
               last_activation=True,
               device=None,
               dtype=None):
    super().__init__()
    assert layers >= 0

    factory_kwargs = {'device': device, 'dtype': dtype}
    self.l_constructor = linear
    self.nonlinearity = non_linearity
    self.in_features = in_features
    self.mid_features = mid_features
    self._results = None
    self._inputs = []
    self._inputs_map = dict()
    self._r_inputs = []
    if layers > 1:
      l = [linear(in_features, mid_features, **factory_kwargs), non_linearity()]
      for _ in range(layers - 2):
        l.extend([linear(mid_features, mid_features, **factory_kwargs), non_linearity()])
      if last_activation:
        l.extend([linear(mid_features, mid_features, **factory_kwargs), non_linearity()])
      else:
        l.append(linear(mid_features, mid_features, **factory_kwargs))
      self.net = nn.Sequential(*l)
    elif layers == 1:
      if last_activation:
        l = [linear(in_features, mid_features, **factory_kwargs), non_linearity()]
      else:
        l = [linear(in_features, mid_features, **factory_kwargs)]
      self.net = nn.Sequential(*l)
    else:
      self.net = lambda x: x
      self.mid_features = self.in_features

    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self._results = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self


  def require_out(self, inputs, out_features: int, purpose: str, bias: bool) -> int:
    if purpose is None:
      name = f"_out_{out_features}_{bias}"
    else:
      name = f"_out_{out_features}_{purpose}_{bias}"
    if not hasattr(self, name):
      self.register_module(name, self.l_constructor(self.mid_features, out_features, bias=bias))
    if name not in self._inputs_map:
      self._inputs_map[name] = []
    input_idx = len(self._inputs)
    self._inputs.append(inputs)
    r_input_idx = len(self._inputs_map[name])
    self._inputs_map[name].append(input_idx)
    self._r_inputs.append((name, r_input_idx))

    return input_idx

  def get_results(self):

    if self._results is None:
      self._results = dict()
      for name, input_map in self._inputs_map.items():
        out_layer = getattr(self, name)
        inputs = torch.stack([self._inputs[i] for i in input_map])
        r = out_layer(inputs)
        self._results[name] = r
    return self._results

  def forward(self, input_id:int, single=False):
    name, r_idx = self._r_inputs[input_id]
    if single:
      out_layer = getattr(self, name)
      return out_layer(self._inputs[input_id])

    result = self.get_results()[name][r_idx]
    return result


class SPredictor(nn.Module):
  def __init__(self,
               out_features: int,
               in_features: int = None,
               shared_net: SimpleMLP = None,
               init_for_task: int = None,
               linear=nn.Linear,
               device=None,
               dtype=None,
               cacheable=DEFAULT_CACHEABLE):
    super().__init__()
    self.cacheable = cacheable
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.init_for_task = init_for_task
    self.in_features = in_features
    self.out_features = out_features
    self.out_parameters = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
    if in_features is None and shared_net is None:
      # not much to do here.
      # They didn't tell us the in_features so we are just predicting the out_features without a sacrificial or shared network
      self.net = None
      self.register_parameter('expansion_data', None)
      self.register_module('in_adapter', None)
      self.register_module('out_adapter', None)
      self.register_parameter('out_adapter_bias', None)

    else:
      if in_features is None:
        in_features = shared_net.in_features
      self.expansion_data = nn.Parameter(torch.empty(in_features, **factory_kwargs))
      # Ok, we do have in parameters so they want us to predict. Do we need to build the net or was one given to us?
      if shared_net is None:
        # We gotta build it
        self.net = linear(in_features, out_features, bias=False, **factory_kwargs)
        net_in_features = in_features
        net_out_features = out_features
        net_has_bias = False
      else:
        self.net = hide_net(shared_net)
        net_in_features = shared_net.in_features
        net_out_features = shared_net.out_features
        net_has_bias = shared_net.has_bias
      if in_features != net_in_features:
        # gotta build an adapter!
        self.in_adapter = nn.Linear(in_features, net_in_features)
      else:
        self.register_module('in_adapter', None)

      if out_features != net_out_features:
        # Gotta build an adapter!
        self.out_adapter = nn.Linear(net_out_features, out_features, bias=False)
        if not net_has_bias:
          # need to create a fake bias for the adapter real quick...
          self.out_adapter_bias = nn.Parameter(torch.empty(net_out_features, **factory_kwargs))
        else:
          self.register_parameter('out_adapter_bias', None)
      else:
        self.register_module('out_adapter', None)
        self.register_parameter('out_adapter_bias', None)

    self.reset_parameters()
    self.cached_value = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_value = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def reset_parameters(self) -> None:
    if not self.out_adapter_bias is None:
      init.uniform_(self.out_adapter_bias, -DEFAULT_BOUND, DEFAULT_BOUND)
    if self.net is None:
      if self.init_for_task is None:
        init.uniform_(self.out_parameters, -DEFAULT_BOUND, DEFAULT_BOUND)
      else:
        init.constant_(self.out_parameters, self.init_for_task)
    else:
      init.uniform_(self.expansion_data, -DEFAULT_BOUND, DEFAULT_BOUND)
      with torch.no_grad():
        v = self._net()
        if self.init_for_task is None:
          # Just do things a bit random
          ift = torch.empty(v.shape).uniform_(-DEFAULT_BOUND, DEFAULT_BOUND)
        else:
          ift = self.init_for_task
        self.out_parameters.set_(ift - v)

  def _net(self):
    x = self.expansion_data
    if not self.in_adapter is None:
      x = self.in_adapter(x)
    x = self.net(x)
    if not self.out_adapter_bias is None:
      # The net they supplied didn't have a bias so add it in
      x = x + self.out_adapter_bias
    if not self.out_adapter is None:
      x = self.out_adapter(x)
    return x

  def forward(self, *args, **kwargs):
    if self.cached_value is None:
      if not self.net is None:
        value = self._net() + (self.out_parameters + self.bias_bias)
      else:
        value = self.out_parameters + self.bias_bias
      if self.cacheable:
        self.cached_value = [value]
    else:
      value = self.cached_value[0]
    return value


class SPredictor2(nn.Module):
  def __init__(self,
               out_features: int,
               shared_net: MultiMLP,
               init_for_task: int = None,
               device=None,
               dtype=None,
               purpose=None,
               cacheable=DEFAULT_CACHEABLE):
    super().__init__()
    self.purpose = purpose
    self.cacheable = cacheable
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.init_for_task = init_for_task
    self.out_features = out_features
    self.out_parameters = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    self.bias_bias = nn.Parameter(torch.zeros(1, **factory_kwargs))

    self.expansion_data = nn.Parameter(torch.empty(shared_net.in_features, **factory_kwargs))
    # Ok, we do have in parameters so they want us to predict. Do we need to build the net or was one given to us?
    self.net = hide_net(shared_net)
    self.net_idx = shared_net.require_out(self.expansion_data, out_features, purpose, bias=False)

    self.reset_parameters()
    self.cached_value = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_value = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def reset_parameters(self) -> None:
    init.uniform_(self.expansion_data, -DEFAULT_BOUND, DEFAULT_BOUND)
    with torch.no_grad():
      v = self.net(self.net_idx, single=True)
      if self.init_for_task is None:
        # Just do things a bit random
        ift = torch.empty(v.shape).uniform_(-DEFAULT_BOUND, DEFAULT_BOUND)
      else:
        ift = self.init_for_task
      self.out_parameters.set_(ift - v)

  def forward(self, *args, **kwargs):
    if self.cached_value is None:
      value = self.net(self.net_idx) + (self.out_parameters + self.bias_bias)
      if self.cacheable:
        self.cached_value = [value]
    else:
      value = self.cached_value[0]
    return value


class SDULinear(nn.Module):
  # Sharable Deep Unified Linear
  def __init__(self,
               in_features: int,
               out_features: int,
               bias=True,
               device=None,
               dtype=None,
               predict_bias=True,
               predict_mbias=True,
               predict_mbias2=True,
               predict_mbias_a=None,
               predict_mbias2_a=None,
               share_in=True,
               share_out=True,
               exp_mul=32.0,
               linear=nn.Linear,
               purpose=None,
               ignore_purpose=True,
               max_predict_size: int = None,
               cacheable=DEFAULT_CACHEABLE) -> None:
    if ignore_purpose:
      self.purpose = None
    else:
      self.purpose = purpose
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.cacheable = cacheable
    self.in_features = in_features
    self.out_features = out_features
    mid_features = min(in_features, out_features)
    # Easy one first
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    expansion_features = int(min(in_features, out_features) * exp_mul)
    if not max_predict_size is None:
      if in_features > max_predict_size:
        # Too big. Gotta cut them down to just parameters
        if predict_mbias == True:
          predict_mbias = False
        if predict_mbias_a == True:
          predict_mbias_a = False
      if out_features > max_predict_size:
        if predict_mbias2 == True:
          predict_mbias2 = False
        if predict_mbias2_a == True:
          predict_mbias2 = False
        if predict_bias == True:
          predict_bias = False

    if share_in == True and (predict_mbias == True or predict_mbias_a == True):
      # Only build this network if we will need it and we are going to use a shared network
      self.in_net = linear(expansion_features, in_features, bias=False, **factory_kwargs)
      in_net = self.in_net
    elif not share_in is None and isinstance(share_in, nn.Module):
      in_net = share_in
      self.register_module('in_net', None)
    else:
      in_net = None
      self.register_module('in_net', None)

    if share_out == True and (predict_mbias2 == True or predict_mbias2_a == True or predict_bias == True):
      # Only build this network if we will need it and we are going to use a shared network
      self.out_net = linear(expansion_features, out_features, bias=False, **factory_kwargs)
      out_net = self.out_net
    elif not share_out is None and isinstance(share_out, nn.Module):
      out_net = share_out
      self.register_module('out_net', None)

    else:
      out_net = None
      self.register_module('out_net', None)

    for name, task, ift in (('mbias', predict_mbias, 1), ('mbias_a', predict_mbias_a, 0)):
      if task is None:
        self.register_module(name, NopModule())
      elif task == True:
        if isinstance(in_net, MultiMLP):
          predictor = SPredictor2(in_features, in_net, init_for_task=ift, purpose=self.purpose, **factory_kwargs)
        else:
          predictor = SPredictor(in_features,
                                 expansion_features,
                                 shared_net=in_net,
                                 init_for_task=ift,
                                 linear=linear,
                                 **factory_kwargs)
        self.register_module(name, predictor)
      else:
        self.register_module(name, SPredictor(in_features,
                                              init_for_task=ift,
                                              **factory_kwargs))
    if bias == False:
      predict_bias = None

    for name, task, ift in (('mbias2', predict_mbias2, 1),
                            ('mbias2_a', predict_mbias2_a, 0),
                            ('bias', predict_bias, None)):
      if task is None:
        self.register_module(name, NopModule())

      elif task == True:
        if isinstance(in_net, MultiMLP):
          predictor = SPredictor2(out_features, out_net, init_for_task=ift, purpose=self.purpose, **factory_kwargs)
        else:
          predictor = SPredictor(out_features,
                                 expansion_features,
                                 shared_net=out_net,
                                 init_for_task=ift,
                                 linear=linear,
                                 **factory_kwargs)
        self.register_module(name, predictor)
      else:
        self.register_module(name, SPredictor(out_features,
                                              init_for_task=ift,
                                              **factory_kwargs))
    self.reset_parameters()
    self.cached_weights = None
    self.register_full_backward_hook(self.clear_cache)

  def clear_cache(self, *args, **kwargs):
    self.cached_weights = None

  def train(self, mode: bool = True):
    super().train(mode)
    self.clear_cache()
    return self

  def eval(self):
    super().eval()
    self.clear_cache()
    return self

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def gen_weights(self) -> (torch.Tensor, torch.Tensor):
    bias = self.bias()
    mbias = self.mbias()
    mbias_a = self.mbias_a()
    mbias2 = self.mbias2()
    mbias2_a = self.mbias2_a()

    if not mbias is None:
      weight = self.weight * mbias
    else:
      weight = self.weight

    if not mbias_a is None:
      weight = weight + mbias_a

    if not mbias2 is None:
      weight = weight.t() * mbias2
      weight = weight.t()

    if not mbias2_a is None:
      weight = weight.t() + mbias2_a
      weight = weight.t()
    return weight, bias

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.cached_weights is None:
      weight, bias = self.gen_weights()
      if self.cacheable:
        self.cached_weights = (weight, bias)
    else:
      weight, bias = self.cached_weights

    result = F.linear(input, weight, bias)
    return result

  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}'


SDULinear = set_accepts_purpose(SDULinear)


class SULinear(nn.Module):
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
