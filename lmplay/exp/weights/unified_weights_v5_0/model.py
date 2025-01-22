import torch
from torch import nn
from typing import Optional, Any, List

from lmplay.exp.weights.modules import ULinear, DULinear
from lmplay.base.encoder.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase, LMRunnerBase
from functools import partial
#See the ULinear in the modules for more info on how this works.

class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6,  # 12 is the real default here
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               version="5.0",
               bias_exp_mul=0.0,
               bias_mid_mul=1.0,
               mbias_exp_mul=8.0,
               mbias_mid_mul=1.0,
               **ignore):
    super().__init__(f"uw_v{version}_{mbias_exp_mul}_{mbias_mid_mul}_{bias_exp_mul}_{bias_mid_mul}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     version=version,
                     **ignore)
    #throwing more parameters! mbias and mbias2 to hit the weights two ways. Totally crazy.
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab

    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    #add in the DULinear to the block definition
    dulinear = partial(DULinear,
                       bias_mid_mul=bias_mid_mul,
                       bias_exp_mul=bias_exp_mul,
                       mbias_mid_mul=mbias_mid_mul,
                       mbias_exp_mul=mbias_exp_mul,
                       mbias2=True,
                       linear=ULinear)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        linear=dulinear) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = dulinear(embed_dim, vocab_size)

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


class ModelRunner(LMRunnerBase):
  def __init__(self, max_batch_size=25):
    super().__init__(max_batch_size=max_batch_size)

  def _construct_model(self,
                       device,
                       model_weights: dict = None,
                       model_args=None,
                       strict=False,
                       **parameters) -> (LMBase, Any):
    model_args = model_args if model_args else dict()
    for k, v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    model = GPT2(**model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
