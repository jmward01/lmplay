import torch
from torch import nn
from typing import Optional, List

from .modules import Block, NNMemory, NNMemoryLayer
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
               nnm_size=128,
               nnm_emb_mul=32,
               nnm_emb_mul2=1,
               nnm_heads=6,
               nnm_ff=True,
               shared_nnm=True,
               shared_nnm_layer=True,
               nnm_first=False,
               extra_nnm_only_blocks=0,
               nnm_attn_residual=True,
               **ignore):
    super().__init__(
      f"{version}_{_p(nnm_ff)}{_p(shared_nnm)}{_p(shared_nnm_layer)}{_p(nnm_first)}{_p(nnm_attn_residual)}_{nnm_size}_{nnm_heads}_{nnm_emb_mul2}_{nnm_emb_mul}_{extra_nnm_only_blocks}_{num_blocks}L_{max_len}",
      max_len=max_len,
      num_heads=num_heads,
      num_blocks=num_blocks,
      embed_dim=embed_dim,
      attn_dropout=attn_dropout,
      ff_dropout=ff_dropout,
      embed_dropout=embed_dropout,
      version=version,
      nnm_size=nnm_size,
      nnm_emb_mul=nnm_emb_mul,
      nnm_heads=nnm_heads,
      nnm_ff=nnm_ff,
      shared_nnm=shared_nnm,
      shared_nnm_layer=shared_nnm_layer,
      nnm_first=nnm_first,
      extra_nnm_only_blocks=extra_nnm_only_blocks,
      nnm_attn_residual=nnm_attn_residual,
      nnm_emb_mul2=nnm_emb_mul2)
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)

    if shared_nnm:
      self.nnm = NNMemory(nnm_size, embed_dim, nnm_heads, emb_mul=nnm_emb_mul2, front_emb_mul=nnm_emb_mul)

      if shared_nnm_layer:
        self.nnm_layer = NNMemoryLayer(self.nnm)
        gen_layer = lambda: self.nnm_layer
      else:
        _layer_count = 0

        def gen_layer():
          nonlocal _layer_count
          nnm_layer = NNMemoryLayer(self.nnm)
          self.register_module(f'nnm_layer_{_layer_count}', nnm_layer)
          _layer_count += 1
          return nnm_layer
    else:
      _layer_count = 0

      def gen_layer():
        nonlocal _layer_count
        nnm = NNMemory(nnm_size, embed_dim, nnm_heads, emb_mul=nnm_emb_mul2, front_emb_mul=nnm_emb_mul)
        nnm_layer = NNMemoryLayer(nnm)
        self.register_module(f'nnm_{_layer_count}', nnm)
        self.register_module(f'nnm_layer_{_layer_count}', nnm_layer)
        _layer_count += 1
        return nnm_layer
    blocks = []
    for i in range(num_blocks):
      b = Block(max_len,
                num_heads,
                embed_dim,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                nnm=gen_layer(),
                nnm_ff=nnm_ff,
                nnm_first=nnm_first,
                nnm_attn_residual=nnm_attn_residual)
      blocks.append(b)
      for _ in range(extra_nnm_only_blocks):
        b = Block(max_len,
                  num_heads,
                  embed_dim,
                  attn_dropout=attn_dropout,
                  ff_dropout=ff_dropout,
                  nnm=gen_layer(),
                  nnm_ff=nnm_ff,
                  nnm_first=nnm_first,
                  nnm_only=True,
                  nnm_attn_residual=nnm_attn_residual)
        blocks.append(b)
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
