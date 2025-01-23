import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

def gen_mask(max_len:int) -> torch.Tensor:
    return  torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)

class MultiheadAttention(nn.Module):
  """This allows testing different ideas. The default behavior (I think) is implemented like the ref transformer block.
  This allows layernorm on kqv and swapping the linear implementation.
  """

  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1, #This was giving me problems. I probably was using it wrong. Maybe I will fix this in the future. For now it is ignored.
               ff_dropout: Optional[float] = 0.1,
               force_ref=False,
               norm_v=False,
               norm_k=False,
               norm_q=False,
               linear=nn.Linear): #Passing in the class we want for a linear layer since this can be swapped for different exp
    """
    :param max_len: Max sequence generation length. Needed for mask generation. Better implementations don't need this.
    :param num_heads: Guess
    :param embed_dim: Guess - must be a multiple of num_heads
    :param attn_dropout: Not implemented. Here mainly as a placeholder to point out it is missing.
    :param ff_dropout: Guess
    :param force_ref: Force the pytorch ref implementation. Only makes a performance difference.
    :param linear: The linear layer class/factory method to instantiate linear layers with.
    """
    # The mask here is generated by the 'casual' value.
    # There are a lot of good reasons to generate a custom mask but for simplicity and clarity this class will deal with mask generation as needed.
    super().__init__()
    assert embed_dim % num_heads == 0, "Embed dim must be a multiple of num_heads."
    self.num_heads = num_heads
    # k&v are what are 'attended' to and will be cached for generation.
    self.key = linear(embed_dim, embed_dim)
    self.value = linear(embed_dim, embed_dim)
    if norm_v:
      self.value_norm = nn.LayerNorm(int(embed_dim / num_heads))
    else:
      self.value_norm = lambda x:x

    if norm_k:
      self.key_norm = nn.LayerNorm(int(embed_dim / num_heads))
    else:
      self.key_norm = lambda x:x

    if norm_q:
      self.query_norm = nn.LayerNorm(int(embed_dim / num_heads))
    else:
      self.query_norm = lambda x:x


    self.query = linear(embed_dim, embed_dim)

    # proj to clean things up after
    self.proj = linear(embed_dim, embed_dim)

    # I had implementation issues with this dropout so I just... dropped it out.
    # It isn't critical to the concept of MHA so this is the easy route.
    # self.attn_dropout = nn.Dropout(attn_dropout)
    self.proj_dropout = nn.Dropout(ff_dropout)

    # not needed for ref version
    self.register_buffer("mask", gen_mask(max_len))
    self.force_ref = force_ref

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

  def forward(self, x, cache: Optional[list] = None) -> torch.Tensor:
    """ Runns mha!
    :param x: How mysterious!
    :param cache: modified in place. The caller shouldn't touch it!
    :return:
    """
    # Useful for later ops
    batch_size, seq_len, embed_dim = x.shape
    if self.force_ref or torch.cuda.is_available():
      k = self.key_norm(self.key(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
      v = self.value_norm(self.value(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
      q = self.query_norm(self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
      # Not adding dropout to match the cpu version
      # y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout, is_causal=self.casual)
      if self._kv_cache_prep(cache):
        k = torch.concat((cache[0], k), dim=2)
        v = torch.concat((cache[1], v), dim=2)
      if cache:
        # Update the cache with the new k&v from this token generation.
        cache[0] = k
        cache[1] = v

      if seq_len > 1:
        # We are either generating the prompt or are in training so we need the casual mask
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
      else:
        # No need to mask since we are just generating the next token.
        # This is broken for batch size > 1 but that is fine for the simple test inference that this project implements.
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
      # This appears to be a little faster on non-cuda hardware for training than the pytorch ref implementation.
      # This also opens up how mha works to make it easier to play with parts of it.

      k = self.key_norm(self.key(x).reshape(batch_size, seq_len, self.num_heads, -1)).permute(0, 2, 3, 1)
      v = self.value_norm(self.value(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
      q = self.query_norm(self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)

      if self._kv_cache_prep(cache):
        k = torch.concat((cache[0], k), dim=-1)
        v = torch.concat((cache[1], v), dim=2)
      if cache:
        # Update the cache with the new k&v from this token generation.
        cache[0] = k
        cache[1] = v

      # This is where the 'scaled' is implemented!
      attn = torch.matmul(q, k) / math.sqrt(q.size(-1))
      # No need to mask for seq len 1 since we are just generating the next token.
      if seq_len > 1:
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))

      # This dropout caused issues on CPU. Didn't spend a lot of time debugging and it isn't critical to the logic of mha.
      # If you are building something that will really be prod ready you should consider forcing ref and adding attn_droupout back
      # or better yet just use a full ref implementation of MHA instead of this class.
      ##attn = self.attn_dropout(attn)
      # attn.shape == (batch_size, num_heads, seq_len, seq_len)
      attn = F.softmax(attn, dim=-1)
      x = torch.matmul(attn, v)

    x = x.transpose(1, 2)
    # y.shape == (batch_size, seq_len, num_heads, head_dim)
    x = x.reshape(batch_size, seq_len, -1)
    # y.shape == (batch_size, seq_len, embed_dim)
    x = self.proj_dropout(self.proj(x))
    return x

class Block(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

  """
  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear, #Passing in the class we want for a linear layer since this can be swapped for different exp
               ln_attn=True,
               ln_mlp=True):
    super().__init__()
    if ln_attn:
      self.ln1 = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x:x
    if ln_mlp:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x:x
    self.attn = MultiheadAttention(max_len,
                                   num_heads,
                                   embed_dim,
                                   attn_dropout=attn_dropout,
                                   ff_dropout=ff_dropout,
                                   linear=linear)
    self.ff = nn.Sequential(linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            linear(embed_dim * 4, embed_dim),
                            nn.Dropout(ff_dropout))

  def forward(self, x, cache:Optional[list]=None):
    #A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    #the 'cache' is the kv cache and is only needed for inference, not training.
    x = x + self.attn(self.ln1(x), cache=cache)
    x = x + self.ff(self.ln2(x))
    return x
