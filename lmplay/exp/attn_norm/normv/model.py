import torch
from torch import nn
from typing import Optional, List

from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase
from lmplay.utils import to_name



class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6, #12 is the real default here
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               start_n=0,
               end_n=None,
               version="1.0",

               **ignore):
    super().__init__(to_name(version, start_n, end_n, num_blocks=num_blocks, max_len=max_len),
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     start_n=start_n,
                     end_n=end_n,
                     version=version)

    if end_n is None:
      end_n =num_blocks

    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        norm_v=i >= start_n and i < end_n,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout) for i in range(num_blocks)])
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
    # This only works for training, not production inference.
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

from lmplay.base.runner_list import expose_runner

