import torch
from torch import nn
from typing import Optional, List

from .modules import Block, FlattenedBatch, FlattenedBatchInfo
import tiktoken
from lmplay.base.base_model import LMBase
from lmplay.utils import to_name


class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6,  #12 is the real default here
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               attn_scales=(3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
               add_attn_position: bool = True,
               add_model_attn: bool = True,
               kv_first:bool = True,
               key_dim=None,
               num_distil_heads=1,
               num_distil_head_groups=None,
               version="1.0",
               **ignore):
    super().__init__(to_name(version,
                             add_model_attn,
                             add_attn_position,
                             kv_first,
                             key_dim,
                             num_distil_heads=num_distil_heads,
                             num_distil_head_groups=num_distil_head_groups,
                             attn_scales=attn_scales,
                             num_blocks=num_blocks,
                             max_len=max_len),
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     key_dim=key_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     attn_scales=attn_scales,
                     version=version,
                     add_model_attn=add_model_attn,
                     add_attn_position=add_attn_position,
                     kv_first=kv_first,
                     num_distil_heads=num_distil_heads,
                     num_distil_head_groups=num_distil_head_groups,
                     expect_extra_loss=True,
                     flat_batch=True,
                     pass_lengths=True)
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    if add_model_attn:
      self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    else:
      self.register_parameter("pos_embed", None)
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(num_heads,
                                        embed_dim,
                                        attn_scales,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        add_position=add_attn_position,
                                        kv_first=kv_first,
                                        key_dim=key_dim,
                                        num_distil_heads=num_distil_heads,
                                        num_distil_head_groups=num_distil_head_groups) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = nn.Linear(embed_dim, vocab_size)

  def forward(self, x:torch.Tensor, cache:Optional[List] = None, lengths:torch.Tensor|None = None):
    if cache is not None and len(cache) > 0:
      raise  NotImplementedError()
      #broken for inference
      #the is part of generate
      #x_start = cache[0][0].size(1)
      #seq_len += x_start
    #assert seq_len <= self.max_len, "sequence longer than model capacity"
    # This only works for training, not production inference.
    tok_embedding = self.tok_embed(x)
    # tok_embedding.shape == (batch_size, seq_len, embed_dim)
    if not self.pos_embed is None:
      #broken for inference
      pos_embedding = torch.concat([self.pos_embed[0, :end, :] for end in lengths], dim=0)
      x = self.dropout(tok_embedding + pos_embedding)
    else:
      #broken for inference
      x = self.dropout(tok_embedding)
    all_attn_loss = 0.0
    x = FlattenedBatch(x, FlattenedBatchInfo(lengths))
    for i, block in enumerate(self.blocks):
      x, attn_loss = block(x, cache=self._kv_cache(cache, i))
      all_attn_loss = all_attn_loss + attn_loss

    x = self.ln(x.data)
    if cache is None:
      #No cache then this is training and we need to decode the whole thing
      x = self.fc(x)
    else:
      #broken for inference
      #Not training. We only care about the last one
      x = self.fc(x[:,-1:,:])
    if not cache is None:
      return x, cache
    all_attn_loss = all_attn_loss/len(self.blocks)
    return x, all_attn_loss