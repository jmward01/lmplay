import torch
from torch import nn
from typing import Optional, Any

from lmplay.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase, LMRunnerBase
from lmplay.modules import SDULinear, ULinear, UnifiedEmbedding, SimpleMLP
from functools import partial


class GPT2(LMBase):
  def __init__(self,
               max_len=1024,
               num_heads=12,
               num_blocks=6,
               embed_dim=768,
               attn_dropout: Optional[float] = 0.1,
               ff_dropout: Optional[float] = 0.1,
               embed_dropout: Optional[float] = 0.1,
               front_embed_mul=16.0,
               exp_mul=64.0,
               for_train=True,
               keep_embed_on_cpu=False,
               ln_attn=False, #UW get a big boost from this and it is fewer parameter/computation so not cheating!
               ln_mlp=False, #UW get a big boost from this and it is fewer parameter/computation so not cheating!
               version="2.0",
               **ignore):
    #Second in the 'sacrificial' line of experiments. These models combine all the sacrificial experiments, experiments that train with extra parameters that are removed for prod.
    #This model could be re-saved after training back to a 'standard' version compatible with the gpt2ish baseline weights.
    #This specific version combines the changes from unified embeddings 1.3 (sort of) and unified weights 2.1
    super().__init__(f"sac_v{version}_{num_blocks}L_{max_len}",
                     max_len=max_len,
                     num_heads=num_heads,
                     num_blocks=num_blocks,
                     embed_dim=embed_dim,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout,
                     embed_dropout=embed_dropout,
                     front_embed_mul=front_embed_mul)

    keep_embed_on_cpu = for_train and keep_embed_on_cpu
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab
    expansion_size = int(exp_mul*embed_dim)
    self.shared_net = SimpleMLP(expansion_size, embed_dim, bias=False)
    self.max_len = max_len
    dulinear = partial(SDULinear,
                       share_in=self.shared_net,
                       share_out=self.shared_net,
                       exp_mul=exp_mul,
                       linear=ULinear)

    self.tok_embed = UnifiedEmbedding(vocab_size,
                                      embed_dim,
                                      front_embed_mul,
                                      keep_embed_on_cpu=keep_embed_on_cpu,
                                      linear=ULinear)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        linear=dulinear,
                                        ln_attn=ln_attn,
                                        ln_mlp=ln_mlp) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.fc = dulinear(embed_dim, vocab_size)

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
    for k,v in parameters.items():
      if k not in model_args:
        model_args[k] = v
    model = GPT2(for_train=self.for_train, **model_args)
    if model_weights is not None:
      missing, unexpected = model.load_state_dict(model_weights, strict=strict)
      model.to(device)
      return model, model.init_kwargs, missing, unexpected
    model.to(device)
    return model, model.init_kwargs
