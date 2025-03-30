import torch
from torch import nn
from typing import Optional

from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase
from lmplay.modules import UnifiedEmbedding, ULinear
import torch.nn.functional as F

def _p(v) -> str:
  if v is None:
    return 'N'
  if isinstance(v, str):
    return v
  return str(int(v))

class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6,
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               front_embed_mul=8.0,
               linear='l',
               tok_embed=True,
               pos_embed=False,
               emb_activation="g",
               mid_mul=1.0,
               norm_v=None,
               for_train=True,
               keep_embed_on_cpu=False,
               version="1.0",
               **ignore):
    super().__init__(f"{version}_{front_embed_mul}_{_p(linear)}{_p(tok_embed)}{_p(pos_embed)}{_p(emb_activation)}{_p(norm_v)}{mid_mul:0.1f}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     front_embed_mul=front_embed_mul,
                     linear=linear,
                     tok_embed=tok_embed,
                     pos_embed=pos_embed,
                     emb_activation=emb_activation,
                     mid_mul=mid_mul)
    keep_embed_on_cpu = for_train and keep_embed_on_cpu
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    #The 'for_train' stuff is boken right now. At some point I'll put it back in a better way.
    #if not for_train:
    #  #this will convert any UE into a normal embedding. After this, if the model is saved, it can be re-loaded by the baseline model.
    #  self.tok_embed = ConvertableEmbedding(vocab_size, embed_dim, front_embed_mul)
    #else:
    if emb_activation == 'g':
      emb_activation = F.gelu
    elif emb_activation == 'r':
      emb_activation = F.relu
    elif emb_activation == 'e':
      emb_activation = F.elu
    else:
      raise ValueError(f"Unknown emb_activation type {emb_activation}")

    if linear == 'l':
      linear = nn.Linear
    elif linear == 'u':
      linear = ULinear
    else:
      raise ValueError(f"Unknown linear type {linear}")
    integration2 = int(embed_dim*mid_mul)
    if tok_embed:
      self.tok_embed = UnifiedEmbedding(vocab_size,
                                        embed_dim,
                                        front_embed_mul,
                                        emb_activation=emb_activation,
                                        integration2=integration2,
                                        linear=linear,
                                        keep_embed_on_cpu=keep_embed_on_cpu)
    else:
      self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    if pos_embed:
      self.pos_embed = UnifiedEmbedding(max_len,
                                        embed_dim,
                                        front_embed_mul,
                                        integration2=integration2,
                                        emb_activation=emb_activation,
                                        linear=linear,
                                        keep_embed_on_cpu=keep_embed_on_cpu)
    else:
      self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    if norm_v is None:
      start_n = 0
      norm_v = False
    else:
      start_n = norm_v
      norm_v = True
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        norm_v=norm_v == True and i >= start_n,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout) for i in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = nn.Linear(embed_dim, vocab_size)

  def forward(self, x:torch.Tensor, cache:Optional[list] = None):
    seq_len = x.size(1)
    x_start = 0
    if cache is not None and len(cache) > 0:
      #the is part of generate
      x_start = cache[0][0].size(1)
      seq_len += x_start
    assert seq_len <= self.max_len, "sequence longer than model capacity"
    tok_embedding = self.tok_embed(x)
    # tok_embedding.shape == (batch_size, seq_len, embed_dim)
    if isinstance(self.pos_embed, UnifiedEmbedding):
      pos_embedding = self.pos_embed(start_slice=x_start, end_slice=seq_len).unsqueeze(0)
    else:
      pos_embedding = self.pos_embed[:, x_start:seq_len, :]
    # pos_embedding.shape == (1, seq_len, embed_dim)
    x = self.dropout(tok_embedding + pos_embedding)
    for i, block in enumerate(self.blocks):
      x = block(x, cache=self._kv_cache(cache, i))
    x = self.ln(x)
    if cache is None:
      #No cache then this is training and we need to decode the whole thing
      x = self.fc(x)
    else:
      #Not training. We only care about the last one
      x = self.fc(x[:,-1:,:])
    if not cache is None:
      return x, cache
    return x


