import torch
from torch import nn
from typing import Optional, List

from .modules import Block as PABlock
from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase


def _p(v) -> str:
  if v is None:
    return 'N'
  return str(int(v))


class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6,  # 12 is the real default here
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               version="1",
               mul=False,  # Mul looks like it takes more mem and works worse in the long run
               final_apos=6,
               **ignore):
    super().__init__(f"{version}_{_p(mul)}_{final_apos}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     version=version,
                     mul=mul,
                     final_apos=final_apos)
    assert final_apos <= num_blocks
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    blocks = [Block(max_len,
                    num_heads,
                    embed_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout) for _ in range(num_blocks - final_apos)]
    blocks.extend([PABlock(max_len,
                           num_heads,
                           embed_dim,
                           attn_dropout=attn_dropout,
                           ff_dropout=ff_dropout,
                           mul=mul) for _ in range(final_apos)])
    self.blocks = nn.Sequential(*blocks)
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = nn.Linear(embed_dim, vocab_size)

  def forward(self, x: torch.Tensor, cache: Optional[List] = None):
    seq_len = x.size(1)
    x_start = 0
    if cache is not None and len(cache) > 0:
      # the is part of generate
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
      # No cache then this is training and we need to decode the whole thing
      x = self.fc(x)
    else:
      # Not training. We only care about the last one
      x = self.fc(x[:, -1:, :])
    if not cache is None:
      return x, cache
    return x
