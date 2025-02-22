import torch
from torch import nn
from typing import Optional

from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase
from lmplay.modules import ULinear
from lmplay.modules import UnifiedEmbedding, ConvertableEmbedding
from lmplay.exp.weights.unified_weights_v2_1.modules import SULinear
from functools import partial

def _p(v) -> str:
  if v is None:
    return 'N'
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
               front_embed_mul=16,
               for_train=True,
               keep_embed_on_cpu=False,
               sulinear=False,
               version="all_1",
               **ignore):
    # First in the 'sacrificial' line of experiments. These models combine all the sacrificial experiments, experiments that train with extra parameters that are removed for prod.
    # This model could be re-saved after training back to a 'standard' version compatible with the gpt2ish baseline weights.
    # This specific version combines the changes from unified embeddings 1.3 and unified weights 1.0
    super().__init__(f"{version}_{front_embed_mul}_{_p(sulinear)}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     front_embed_mul=front_embed_mul)
    # same as 1.0 but the linear is now a ULinear!
    keep_embed_on_cpu = for_train and keep_embed_on_cpu
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab
    if sulinear:
      self.shared_mid_weights = nn.Linear(embed_dim, embed_dim)
      #This version combines UW 1.0 and UW 2.0
      linear = partial(SULinear, self.shared_mid_weights)
    else:
      linear = ULinear

    self.max_len = max_len
    if not for_train:
      # this will convert any UE into a normal embedding. After this, if the model is saved, it can be re-loaded by the baseline model.
      self.tok_embed = ConvertableEmbedding(vocab_size, embed_dim, front_embed_mul)
    else:
      self.tok_embed = UnifiedEmbedding(vocab_size, embed_dim, front_embed_mul, keep_embed_on_cpu=keep_embed_on_cpu,
                                        linear=linear)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        linear=linear,
                                        mha_lradd=True,
                                        ff_lradd=True) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = linear(embed_dim, vocab_size)

  def forward(self, x: torch.Tensor, cache: Optional[list] = None):
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
