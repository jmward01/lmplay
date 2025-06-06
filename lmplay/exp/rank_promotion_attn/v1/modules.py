import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F, init
from torch.nn.utils import rnn

__all__ = ['DistiledMultiheadAttention']


# stolen from pytorch docs and modified.
def scaled_dot_product_attention(query, key, value, num_heads: int, dropout_p=0.0, training: bool = True) -> (
torch.Tensor, torch.Tensor):
  target_batch_size, target_seq_len, target_embed_dim = value.shape
  key = key.reshape(target_batch_size, target_seq_len, num_heads, -1).transpose(1, 2)
  value = value.reshape(target_batch_size, target_seq_len, num_heads, -1).transpose(1, 2)
  query = query.reshape(target_batch_size, 1, num_heads, -1).transpose(1, 2)

  scale_factor = 1 / math.sqrt(query.size(-1))
  attn_weight = query @ key.transpose(-2, -1) * scale_factor
  attn_weight = torch.softmax(attn_weight, dim=-1)
  if dropout_p > 0 and training == True:
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  result = attn_weight @ value
  result = result.view(target_batch_size, -1)
  attn_weight = attn_weight.detach().view(target_batch_size, num_heads, target_seq_len)
  attn_weight = torch.mean(attn_weight, dim=1)
  return result, attn_weight


class MHA(nn.Module):
  def __init__(self, embedding_dim: int, num_heads: int, k_dim: int | None = None, dropout: float = 0.0):
    super().__init__()
    if k_dim is None:
      k_dim = embedding_dim
    self.num_heads = num_heads
    self.k_dim = k_dim
    self.embedding_dim = embedding_dim
    self.k_proj = nn.Linear(k_dim, k_dim)
    self.q_proj = nn.Linear(embedding_dim, k_dim)
    self.v_proj = nn.Linear(embedding_dim, embedding_dim)
    self.dropout = dropout

  def forward(self, q, k, v):
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    return scaled_dot_product_attention(q, k, v, self.num_heads, dropout_p=self.dropout, training=self.training)


def roll_by_gather(data, shifts: torch.Tensor, guard_mask=None):
  # modified from https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch
  # The guard mask works for this specific use where we are shifting indexes to tile in a buffer.
  # We don't want the tiled buffer idexs coming back anyway since that would waste memory and without them there is no need
  # to do the mod so this saves mem and computation.
  n_rows, n_cols = data.shape[:2]
  arange1 = torch.arange(n_rows, device=data.device).view((n_rows, 1)).repeat((1, n_cols))
  if guard_mask is None:
    arange2 = (arange1 - shifts) % n_rows
  else:
    arange2 = (arange1 - shifts)
    arange2 = arange2[guard_mask]
  return torch.gather(data, 0, arange2)


def tile_within(x, buffer):
  # Alternate/faster version: https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch
  batch_size, seq_length, embed_dim = x.shape
  buffer = buffer.expand(batch_size, -1, -1)
  _, scale_window_length, embed_dim = buffer.shape

  scale_window_length += 1
  new_layer = []
  proto_layer = torch.concat([buffer, x], dim=1)
  for i in range(scale_window_length):
    offset = (scale_window_length - i) - 1
    end = offset + seq_length
    new_layer.append(proto_layer[:, offset:end])
  new_layer = torch.stack(new_layer, dim=2)
  return new_layer


class FlattenedBatchInfo:
  def __init__(self, sample_lengths: torch.Tensor):
    self.sample_lengths = sample_lengths
    self._sample_lengths_list: list[int] | None = None
    self._max_sample_length: int | None = None
    self._sample_start_idxs: torch.Tensor | None = None
    self._flat_size: int | None = None

  @property
  def sample_lengths_list(self) -> list[int]:
    if self._sample_lengths_list is None:
      self._sample_lengths_list = self.sample_lengths.tolist()
    return self._sample_lengths_list

  @property
  def max_sample_length(self) -> int:
    if self._max_sample_length is None:
      self._max_sample_length = max(self.sample_lengths_list)
    return self._max_sample_length

  @property
  def sample_start_idxs(self):
    if self._sample_start_idxs is None:
      self._sample_start_idxs = torch.concat([torch.tensor([0], dtype=torch.int64, device=self.sample_lengths.device),
                                              torch.cumsum(self.sample_lengths, 0, dtype=torch.int64)[:-1]])
    return self._sample_start_idxs

  @property
  def batch_size(self) -> int:
    return len(self.sample_lengths_list)

  @property
  def flat_size(self) -> int:
    if self._flat_size is None:
      self._flat_size = sum(self.sample_lengths_list)
    return self._flat_size

  def tile_within(self, data: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
    # Buffer size dictates the tile size. Tile = buffer.shape[0] + 1
    # buffer: len, emb_size
    buff_len = buffer.shape[0]
    seq_idxs = torch.arange(buff_len, buff_len + data.shape[0], device=data.device)
    buf_idxs = torch.arange(0, buff_len, device=data.device)
    last_idx = 0
    idx_list = []
    # Not a fan of this loop but it should be relatively fast since it is batch size long.
    # This should be dominated by all other computation. Still. This is probably one of the slowest parts of this function.
    for next_idx in self.sample_lengths_list:
      idx_list.append(buf_idxs)
      idx_list.append(seq_idxs[last_idx:last_idx + next_idx])
      last_idx += next_idx

    # This list of idxs now has every sample idx in it withpadding idxs prefixing them. Whe we tile this it will give us the almost idxs we need.
    tile_idxs = torch.concat(idx_list)
    # This will allow us to select out the tiled areas we care about only
    data_idxs = tile_idxs >= buff_len
    # we will roll to create tiling of the source data and the buffer absorbs the stuff that went into negative indexes
    shift = torch.arange(0, buff_len + 1, device=buffer.device).unsqueeze(0)
    tile_idxs = tile_idxs.unsqueeze(-1).expand(-1, buff_len + 1)
    tile_idxs = roll_by_gather(tile_idxs, shift, guard_mask=data_idxs)
    tile_idxs = tile_idxs.reshape(-1)
    tiled_data = torch.concat([buffer, data])[tile_idxs]

    tiled_data = tiled_data.view(data.shape[0], -1, *data.shape[1:])
    return tiled_data

  def unflatten(self, data) -> torch.Tensor:
    # Turn this back into a batch. I assume the pytorch routines here are fast.
    tensors = list(torch.split(data, self.sample_lengths_list))
    fp = rnn.pack_sequence(tensors, False)
    d, lengths = rnn.pad_packed_sequence(fp, batch_first=True)
    return d

  @classmethod
  def flatten(cls, batch, sample_lengths: torch.Tensor) -> (torch.Tensor, "FlattenedBatchInfo"):
    # Not a huge fan of this method. I wish the pytorch rnn stuff was a little easier to deal with.
    # It probably is, I just am missing how they pack things in a PackedSequence to be able to easily use it.
    parts = []
    for sample, length in zip(batch, sample_lengths):
      parts.append(sample[:length])
    return torch.concat(parts), cls(sample_lengths)


class FlattenedBatch:
  def __init__(self, data: torch.Tensor, lengths: FlattenedBatchInfo):
    self.data = data
    self.lengths = lengths
    # Useful for debugging
    assert data.shape[0] == lengths.flat_size

  @property
  def sample_lengths(self) -> torch.Tensor:
    return self.lengths.sample_lengths

  @property
  def sample_lengths_list(self) -> list[int]:
    return self.lengths.sample_lengths_list

  @property
  def max_sample_length(self) -> int:
    return self.lengths.max_sample_length

  @property
  def sample_start_idxs(self):
    return self.lengths.sample_start_idxs

  @property
  def batch_size(self) -> int:
    return self.lengths.batch_size

  def tile_within(self, buffer: torch.Tensor) -> "FlattenedBatch":
    tile_data = self.lengths.tile_within(self.data, buffer)
    return FlattenedBatch(tile_data, self.lengths)

  def unflatten(self) -> torch.Tensor:
    return self.lengths.unflatten(self.data)

  @classmethod
  def flatten(cls, batch, sample_lengths: torch.Tensor) -> "FlattenedBatch":
    data, lengths = FlattenedBatchInfo.flatten(batch, sample_lengths)
    return FlattenedBatch(data, lengths)


class DistiledMultiheadAttention(nn.Module):
  """Trying ideas to reduce the overall attn length
  """

  def __init__(self,
               scale_lengths: list[int],
               num_heads: int,
               embed_dim: int,
               num_distil_heads:int=1,
               num_distil_head_groups=None,
               key_dim: int | None = None,
               ff_dropout: Optional[float] = 0.1,
               add_position: bool = True,
               kv_first: bool = True,
               **kwargs):
    super().__init__()

    assert embed_dim % num_heads == 0, "Embed dim must be a multiple of num_heads."
    self.num_distil_heads = num_distil_heads
    if num_distil_head_groups is None:
      num_distil_head_groups = self.num_distil_heads
    self.num_distil_head_groups = num_distil_head_groups
    self.num_heads = num_heads
    self.emb_dim = embed_dim
    if key_dim is None:
      key_dim = self.emb_dim
    self.key_dim = key_dim
    # head_size = int(embed_dim / num_heads)

    # k&v are what are 'attended' to and will be cached for generation.
    # Tying them together for performance
    self.kv_first = kv_first
    if kv_first:
      scale_dim = embed_dim + self.key_dim
    else:
      scale_dim = embed_dim
    post_tile_dim = embed_dim + self.key_dim
    self.key_value = nn.Linear(embed_dim, post_tile_dim)

    self.query = nn.Linear(embed_dim, key_dim)
    # proj to clean things up after
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.proj_dropout = nn.Dropout(ff_dropout)

    # A scale length is the number of elements dedicated to that scale level. sum(*scale_lengths) = context length
    self.scale_window_lengths = scale_lengths
    scale_distilations = []
    utility_predictors = []
    layer_buffers = []

    def build_scale_distilation():
      l1 = nn.Linear(scale_length * scale_dim, scale_length * 10)
      l2 = nn.Linear(scale_length * 10, scale_length*num_distil_heads)
      return nn.Sequential(l1, nn.GELU(), l2)

    def build_utility_prediction():
      l1 = nn.Linear(scale_length * scale_dim, scale_length * 10)
      l2 = nn.Linear(scale_length * 10, 1)
      return nn.Sequential(l1, nn.GELU(), l2)

    # Don't need to distil or predict the last layer anymore
    for scale_length in scale_lengths[:-1]:
      # Keeping this simple right now.
      # This is predicting the value of each value in the scale window so we can softmax and sum on it.
      scale_distilations.append(build_scale_distilation())
      # The tulity predictor is pretty lightweight since it only outputs size 1
      utility_predictors.append(build_utility_prediction())
      # We need a buffer for each layer at the beginning so that we can stack things properly
      layer_buffers.append(nn.Parameter(torch.empty((scale_length - 1, scale_dim))))
    # The final scale doesn't need predictors for the next level
    layer_buffers.append(nn.Parameter(torch.empty((scale_lengths[-1] - 1, scale_dim))))
    # These construct the next layer
    self.scale_distilations = nn.ModuleList(scale_distilations)
    # These predict the utility of the next layer
    self.utility_predictors = nn.ModuleList(utility_predictors)
    self.layer_buffers = nn.ParameterList(layer_buffers)
    if add_position:
      self.position = nn.Parameter(torch.zeros(1, sum(self.scale_window_lengths), post_tile_dim))
    else:
      self.register_parameter("position", None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for p in self.layer_buffers:
      init.uniform_(p)

  def _kv_cache_prep(self, cache: Optional[list]) -> bool:
    """ preps the cache with 'None' so that future sets have space.
    :param cache:
    :return: if the cache has a value that should be appended to.
    """
    if cache is None:
      return False
    if len(cache) == 0:
      cache.extend([None, None])
      return False
    return True

  def _gen_attn_layer(self, f_x: FlattenedBatch, layer: int):
    # First things first, lets stack things up on this layer.
    tiled_f_x = f_x.tile_within(self.layer_buffers[layer])

    # Gen the expected utility
    # Add 1 because it can't go below 0
    expected_utility = F.elu(self.utility_predictors[layer](tiled_f_x.data.view(tiled_f_x.data.shape[0], -1))) + 1.0
    expected_utility = expected_utility.squeeze(-1)
    # batch, seq_len
    # Now we generate a running average of expected utility
    # This isn't quite right yet since this average is running across samples. We will fix that later
    with torch.no_grad():
      # No gradient needed here.
      # This logic will be learend purely based on the predicted utility matching the actual utility use.
      running_ave_utility = torch.cumsum(expected_utility, 0)
      end_idxs = f_x.sample_start_idxs + f_x.sample_lengths - 1
      end_utility = running_ave_utility[end_idxs]
      # This gives us the utility for all samples in the batch except for the last
      first_utility = end_utility[0:1]
      if f_x.sample_lengths.shape[0] > 1:
        # If we have a batch > 1 we need to adjust all the running stuff since cumsum takes into account the other smaples
        all_but_first_utility = end_utility[2:] - end_utility[1:-1]
        sample_utility = torch.concat([first_utility, all_but_first_utility])

        # now we do a little cheat
        z = torch.zeros_like(running_ave_utility)
        z[f_x.sample_start_idxs[1:]] = sample_utility
        z = torch.cumsum(z, 0)
        # now we have the utility running average not impacted by the previous batch values
        # Let's do something similar for the index of every sample
        running_ave_utility -= z
        z = torch.zeros_like(running_ave_utility)
        z[f_x.sample_start_idxs[1:]] = f_x.sample_lengths[:-1].to(z.dtype)
        z = torch.cumsum(z, 0)
        idxs = torch.arange(1, f_x.data.shape[0] + 1, device=z.device)
        idxs = idxs - z
      else:
        idxs = torch.arange(1, f_x.data.shape[0] + 1, device=f_x.data.device)
      running_ave_utility = running_ave_utility / idxs
      # We will always take until our scale is filled so set those appropriately
      required_idxs = idxs <= self.scale_window_lengths[layer]
      selected = expected_utility > running_ave_utility
      selected = torch.logical_or(selected, required_idxs, out=selected)

    # We need to know how long the new sample lengths are. To do that we count up the selected and use that to find the new lengths
    cumsum = torch.cumsum(selected, 0)
    end_counts = cumsum[end_idxs]
    next_layer_lengths = torch.concat((end_counts[0:1], end_counts[1:] - end_counts[:-1]))
    # Now we know how long each sample in the new scale layer will be

    # This builds a mapping that will take the selected and expand it to the same size as the original input.
    # This is because all the 'false' values in the cumsum come up as consecutive. this gives us the last true indexes for all those false ones.
    _, selected_take_map = torch.unique_consecutive(cumsum, dim=None, return_inverse=True)

    # This will be used to generate loss
    selected_expected_utility = expected_utility[selected]

    # Now we need to create the promoted values
    # Using a simple FF to create a distilled new value
    # The result of new_layer_big[selected] is a flattened array because the number selected for each part of the batch can be different.
    selected_tiled_f_x = tiled_f_x.data[selected]
    flat_seq, scale_window_len = selected_tiled_f_x.shape[:2]
    # flatten it and get the rankings
    rankings = self.scale_distilations[layer](selected_tiled_f_x.view(selected_tiled_f_x.shape[0], -1))

    #ranking: flat_seq, scale_window_len*distil_heads
    rankings = rankings.view(flat_seq, self.num_distil_heads, 1, scale_window_len)
    #ranking: flat_seq, distil_heads, 1 , scale_window_len

    #rankings   = flat_seq, scale_window
    #selected_tiled_f_x = flat_seq, scale_window, tile_size
    #We need
    #rankings   = flat_seq, heads, 1, scale_window
    #selected_tiled_f_x = flat_seq, heads, scale_window, tile_size/heads
    rankings = torch.softmax(rankings, -1)

    if self.num_distil_heads != self.num_distil_head_groups:
      rankings = rankings.view(flat_seq,
                               self.num_distil_head_groups,
                               int(self.num_distil_heads/self.num_distil_head_groups),
                               scale_window_len)
      rankings = torch.sum(rankings, dim = 2).unsqueeze(-2) / int(self.num_distil_heads/self.num_distil_head_groups)
    selected_tiled_f_x = selected_tiled_f_x.view(flat_seq, scale_window_len, self.num_distil_head_groups, -1)
    selected_tiled_f_x = selected_tiled_f_x.permute(0, 2, 1, 3)

    next_layer = rankings @ selected_tiled_f_x
    next_layer = next_layer.view(flat_seq, -1)

    #next_layer = selected_tiled_f_x * rankings
    #next_layer = torch.sum(next_layer, -2)

    next_layer = FlattenedBatch(next_layer, FlattenedBatchInfo(next_layer_lengths))

    selected_expected_utility = FlattenedBatch(selected_expected_utility.unsqueeze(-1), next_layer.lengths)
    utility_buffer = torch.zeros((self.scale_window_lengths[layer + 1] - 1, 1),
                                 device=selected_expected_utility.data.device)
    # This utility will map to the tiled next layer and selected for the previous layer so just do it now.
    selected_expected_utility = selected_expected_utility.tile_within(utility_buffer)
    return tiled_f_x, next_layer, selected_take_map, selected_expected_utility.data

  def forward(self, x: FlattenedBatch, cache: Optional[list] = None) -> (torch.Tensor, torch.Tensor):
    # Lengths are always passed right now since we don't support prod inferrence yet.

    # Useful for later ops
    # batch_size, seq_len, embed_dim = x.shape

    # Start by flattening it all out
    # x = FlattenedBatch.flatten(x, lengths)
    lengths = x.lengths
    if self.kv_first:
      current_layer = FlattenedBatch(self.key_value(x.data), x.lengths)
    else:
      current_layer = FlattenedBatch(x.data, x.lengths)

    # We need to save things each round to put it all back together
    things_to_save = []

    for i in range(len(self.scale_window_lengths) - 1):
      tiled_previous_f_x, next_layer, selected_take_map, selected_expected_utility = self._gen_attn_layer(current_layer,
                                                                                                          i)
      current_layer = next_layer
      # selected_take_map will take the next layer and fill it to be the same length as the previous layer
      things_to_save.append((tiled_previous_f_x, selected_take_map, selected_expected_utility))
      # tiled_layer = tile_within(current_layer)
      # selected_fb = fb of the values selected from the untiled passed in current_layer
      # expected_utility_fb = expected utility mapped to the passed back selected_fb

    # current_layer now equals the final scale layer. We need to tile that so we can use it later.
    last_kv = current_layer.tile_within(self.layer_buffers[-1]).data

    # Track the last layer and start the takemap for it
    # take_maps = [[torch.arange(last_kv.data.shape[0], dtype=torch.int64, device=last_kv.data.device)]]
    take_maps = []
    for tiled_previous_f_x, selected_take_map, selected_expected_utility in reversed(things_to_save):
      take_maps.append(
        [torch.arange(last_kv.shape[0], dtype=torch.int64, device=last_kv.device), last_kv, selected_expected_utility])
      for d in take_maps:
        d[0] = d[0][selected_take_map]
      last_kv = tiled_previous_f_x.data

    expected_utilities = []
    scale_histories = []

    for take_map, scale_history, expected_utility in take_maps:
      if self.kv_first:
        scale_histories.insert(0, scale_history[take_map])
      else:
        scale_histories.insert(0, self.key_value(scale_history)[take_map])
      expected_utilities.insert(0, expected_utility[take_map])
    if self.kv_first:
      scale_histories.insert(0, last_kv)
    else:
      scale_histories.insert(0, self.key_value(last_kv))
    kv = torch.concat(scale_histories, dim=-2)
    expected_utility = torch.concat(expected_utilities, dim=-2)
    q = self.query(x.data).view(-1, 1, self.key_dim)
    # q: seq_len, 1, num_heads, head_size
    if not self.position is None:
      kv = kv + self.position
    k = kv[:, :, :self.key_dim]
    v = kv[:, :, self.key_dim:]
    x, utility = scaled_dot_product_attention(q, k, v, self.num_heads)
    # x, utility = self.mha(q,k,v)
    # x = x.squeeze(-2)

    expected_utility = expected_utility.squeeze(-1)
    # utility = utility.squeeze(-2)[:,self.scale_window_lengths[0]:].detach()
    utility = utility[:, self.scale_window_lengths[0]:].detach()
    u_map = expected_utility > 0
    utility = utility[u_map]
    expected_utility = expected_utility[u_map]
    x = self.proj_dropout(self.proj(x))
    utility_loss = F.mse_loss(expected_utility, utility.detach(), reduction="mean")
    x = FlattenedBatch(x, lengths)
    return x, utility_loss


from torch import nn
from typing import Optional

from lmplay.modules.general import LRAdd
from lmplay.utils import create_linear


class Block(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

  """

  def __init__(self,
               num_heads: int,
               embed_dim: int,
               scale_lengths: list[int],
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear,
               # Passing in the class we want for a linear layer since this can be swapped for different exp
               ff_linear=None,
               lradd=False,
               lradd_simple=None,  # only matters is lradd=True
               lradd_predict=None,  # only matters is lradd=True
               lradd_floor=None,  # only matters is lradd=True
               lradd_ceil=None,  # only matters is lradd=True
               ln_attn=True,
               ln_mlp=True,
               key_dim=None,
               **kwargs):
    super().__init__()
    if ff_linear is None:
      ff_linear = linear
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x: x
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x: x
    if lradd:
      self.ff_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.ff_lradd = lambda x, y: x + y

    if lradd:
      self.mha_lradd = LRAdd(embed_dim, simple=lradd_simple, predict=lradd_predict, floor=lradd_floor, ceil=lradd_ceil)
    else:
      self.mha_lradd = lambda x, y: x + y

    self.attn = DistiledMultiheadAttention(scale_lengths,
                                           num_heads,
                                           embed_dim,
                                           ff_dropout=ff_dropout,
                                           key_dim=key_dim,
                                           **kwargs)
    self.ff = nn.Sequential(create_linear(ff_linear, 'block_ff_1', embed_dim, embed_dim * 4),
                            nn.GELU(),
                            create_linear(ff_linear, 'block_ff_2', embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))

  def forward(self, x: FlattenedBatch, cache: Optional[list] = None):
    lengths = x.lengths
    x = x.data
    # A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    # the 'cache' is the kv cache and is only needed for inference, not training.
    attn, attn_loss = self.attn(FlattenedBatch(self.ln1(x), lengths), cache=cache)
    x = self.mha_lradd(x, attn.data)
    x = self.ff_lradd(x, self.ff(self.ln2(x)))
    return FlattenedBatch(x, lengths), attn_loss
