import torch
from torch import nn
from typing import Optional, List

from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase, MBase
from lmplay.base.base_recurrent_model import RLMBase


class _GPT2(MBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6, #12 is the real default here
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               basename="GPT2ish",
               **ignore):
    super().__init__(f"{basename}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout)
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = nn.Linear(embed_dim, vocab_size)

  def forward(self, x:torch.Tensor, cache:Optional[List] = None):
    seq_len = x.size(1)
    x_start = 0
    if cache is not None and len(cache) > 0:
      #the is part of generate
      x_start = cache[0][0].size(1)
      seq_len += x_start
    assert seq_len <= self.max_len, "sequence longer than model capacity"
    tok_embedding = self.tok_embed(x)
    # tok_embedding.shape == (batch_size, seq_len, embed_dim)
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

class GPT2(_GPT2, LMBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class RGPT2(_GPT2, RLMBase):
  """Just here to have a ref implementation of recurrent that should perform exactly like GPT2

  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, basename="RGPT2ish", **kwargs)

  def forward(self, x:torch.Tensor, s:torch.Tensor, cache:List):
    x, cache = super().forward(x, cache=cache)
    return x, cache, None