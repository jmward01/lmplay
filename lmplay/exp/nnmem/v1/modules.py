import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from lmplay.utils import create_linear
from lmplay.modules import MultiheadAttention, UnifiedEmbedding
import math
from torch.nn import init

class CachedOutput(nn.Module):
  def __init__(self, module:nn.Module):
    super().__init__()
    self.module = [module]
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

  def forward(self, *args, **kwargs):
    if self.cached_value is None:
      self.cached_value = self.module[0]()
    return self.cached_value

class NNEmbedding(nn.Module):
  def __init__(self, cells, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.weight = nn.Parameter(torch.empty((cells, embedding_dim)))
    init.normal_(self.weight)

  def forward(self):
    return self.weight

class NNMemory(nn.Module):
  def __init__(self,
               cells:int,
               embedding_dim:int,
               num_heads:int,
               front_emb_mul=64,
               emb_mul=1,
               linear=nn.Linear,
               force_ref = False):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.force_ref = force_ref
    nnm_embedding = int(emb_mul*embedding_dim)
    if front_emb_mul == 0:
      self.nnm = NNEmbedding(cells, nnm_embedding)
    else:
      self.nnm = UnifiedEmbedding(cells, nnm_embedding, front_embed_mul=front_emb_mul)
    assert embedding_dim % num_heads == 0, "Embed dim must be a multiple of num_heads."
    self.num_heads = num_heads
    # k&v are what are 'attended' to and will be cached for generation.
    #In a prod version these go away completely and turn into fixed K/V paramteres
    self.key = create_linear(linear, 'nnm_key', nnm_embedding, embedding_dim)
    self.value = create_linear(linear, 'nnm_value', nnm_embedding, embedding_dim)
    #So that we don't recalc the k/v parameters every time
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

  def get_KV(self):
    if self.cached_value is None:
      nnm = self.nnm().unsqueeze(0)
      target_batch_size, target_seq_len, target_embed_dim = nnm.shape
      k = self.key(nnm).reshape(target_batch_size, target_seq_len, self.num_heads, -1)
      v = self.value(nnm).reshape(target_batch_size, target_seq_len, self.num_heads, -1).transpose(1, 2)

      self.cached_value = [k,v]
    return self.cached_value[0], self.cached_value[1]

  def forward(self, x):
    # Useful for later ops
    batch_size, seq_len, embed_dim = x.shape
    k, v = self.get_KV()
    q = x.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    if self.force_ref or torch.cuda.is_available():
      k = k.transpose(1, 2)
      x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
      # This also opens up how mha works to make it easier to play with parts of it.
      #It may also be slightly faster than the pytorch implementation on cpu? At least it was at one point.
      k = k.permute(0, 2, 3, 1)
      # This is where the 'scaled' is implemented!
      attn = torch.matmul(q, k) / math.sqrt(q.size(-1))
      attn = F.softmax(attn, dim=-1)
      x = torch.matmul(attn, v)

    x = x.transpose(1, 2)
    # y.shape == (batch_size, seq_len, num_heads, head_dim)
    x = x.reshape(batch_size, seq_len, -1)
    # y.shape == (batch_size, seq_len, embed_dim)
    return x

class NNMemoryLayer(nn.Module):
  def __init__(self,
               nnm:NNMemory,
               linear = nn.Linear,
               proj_dropout: Optional[float] = 0.1):
    super().__init__()
    self. nnm = [nnm]
    self.value = create_linear(linear, 'nnm_q', nnm.embedding_dim, nnm.embedding_dim)
    self.proj = create_linear(linear, 'nnm_proj', nnm.embedding_dim, nnm.embedding_dim)
    self.proj_dropout = nn.Dropout(proj_dropout)

  def forward(self, v):
    v = self.value(v)
    v = self.nnm[0](v)
    v = self.proj_dropout(self.proj(v))
    return v

class Block(nn.Module):
  """Your basic encoder block implementation! Nothing crazy in here.

  """
  def __init__(self,
               max_len: int,
               num_heads: int,
               embed_dim: int,
               nnm:NNMemory|NNMemoryLayer,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               linear=nn.Linear,  #Passing in the class we want for a linear layer since this can be swapped for different exp
               ln_attn=True,
               ln_mlp=True,
               nnm_ff=True,
               nnm_first=False,
               nnm_only=False,
               nnm_attn_residual=True):
    super().__init__()
    self.nnm_first = nnm_first
    self.nnm_attn_residual = nnm_attn_residual
    if ln_attn:
      if not nnm_only:
        self.ln1 = nn.LayerNorm(embed_dim)
      else:
        self.ln1 = lambda x:x
      self.ln1_nnm = nn.LayerNorm(embed_dim)
    else:
      self.ln1 = lambda x:x
      self.ln1_nnm = lambda x:x
    if ln_mlp and not nnm_only:
      self.ln2 = nn.LayerNorm(embed_dim)
    else:
      self.ln2 = lambda x:x

    if nnm_ff:
      self.ln2_nnm = nn.LayerNorm(embed_dim)
      self.ff_nnm = nn.Sequential(create_linear(linear, 'block_ff_nnm_1', embed_dim, embed_dim * 4),
                              nn.GELU(),
                              create_linear(linear, 'block_ff_nnm_2', embed_dim * 4, embed_dim),
                              nn.Dropout(ff_dropout))

    else:
      self.ln2_nnm = lambda x:x
      self.register_module('ff_nnm', None)
    self.nnm = [nnm]

    if not nnm_only:
      self.attn = MultiheadAttention(max_len,
                                     num_heads,
                                     embed_dim,
                                     attn_dropout=attn_dropout,
                                     ff_dropout=ff_dropout,
                                     linear=linear)


      self.ff = nn.Sequential(create_linear(linear, 'block_ff_1', embed_dim, embed_dim * 4),
                              nn.GELU(),
                              create_linear(linear, 'block_ff_2', embed_dim * 4, embed_dim),
                              nn.Dropout(ff_dropout))
    else:
      self.register_module('attn', None)
      self.register_module('ff', None)



  def forward(self, x, cache:Optional[list]=None):
    #A simple 'block' that uses residual connections and gives attn + pure logic both a chance to modify the hidden layer
    #the 'cache' is the kv cache and is only needed for inference, not training.

    if self.nnm_first:
      if self.nnm_attn_residual:
        r = x + self.nnm[0](self.ln1_nnm(x))
        x = r
      else:
        r = self.nnm[0](self.ln1_nnm(x))
      if not self.ff_nnm is None:
        x = x + self.ff_nnm(self.ln2_nnm(r))
      else:
        x = r
      if not self.attn is None:
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.ff(self.ln2(x))
    else:
      if not self.attn is None:
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.ff(self.ln2(x))

      if self.nnm_attn_residual:
        r = x + self.nnm[0](self.ln1_nnm(x))
        x = r
      else:
        r = self.nnm[0](self.ln1_nnm(x))
      if not self.ff_nnm is None:
        x = x + self.ff_nnm(self.ln2_nnm(r))
      else:
        x = r
    return x
