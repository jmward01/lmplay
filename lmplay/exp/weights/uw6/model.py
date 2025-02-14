import torch
from torch import nn
from typing import Optional, List

from lmplay.exp.weights.modules import SDULinear, SimpleMLP, ULinear, MultiMLP, accepts_purpose
from lmplay.base.encoder.modules import Block
import tiktoken
from lmplay.base.base_model import LMBase
from functools import partial


# See the ULinear in the modules for more info on how this works.

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
               version="6",
               exp_mul=16,
               predict_bias=True,
               predict_mbias=True,
               predict_mbias2=True,
               predict_mbias_a=None,  # Close to stable but not quite.
               predict_mbias2_a=None,  # Close to stable but not quite.
               ln_attn=False,
               ln_mlp=False,
               ln_fc=True,
               dl_fc=True,
               share_in=True,
               share_out=True,
               ulinear=False,
               cacheable=True,
               share_layers=2,
               share_mid_mul=4,
               mmlp=False,
               last_activation=True,
               ignore_purpose=True,
               **ignore):
    super().__init__(
      f"uw_v{version}_{_p(predict_bias)}{_p(predict_mbias)}{_p(predict_mbias2)}{_p(predict_mbias_a)}{_p(predict_mbias2_a)}{_p(ln_attn)}{_p(ln_mlp)}{_p(ln_fc)}{_p(dl_fc)}{_p(share_in)}{_p(share_out)}{_p(ulinear)}{_p(cacheable)}{_p(mmlp)}{_p(last_activation)}{_p(ignore_purpose)}_{share_layers}_{share_mid_mul}_{exp_mul}_{num_blocks}L_{max_len}",
      max_len=max_len,
      num_heads=num_heads,
      num_blocks=num_blocks,  # 12 is the real default here
      embed_dim=embed_dim,
      attn_dropout=attn_dropout,
      ff_dropout=ff_dropout,
      embed_dropout=embed_dropout,
      version=version,
      exp_mul=exp_mul,
      predict_bias=predict_bias,
      predict_mbias=predict_mbias,
      predict_mbias2=predict_mbias2,
      predict_mbias_a=predict_mbias_a,
      predict_mbias2_a=predict_mbias2_a,
      ln_attn=ln_attn,
      ln_mlp=ln_mlp,
      ln_fc=ln_fc,
      dl_fc=dl_fc,
      share_in=share_in,
      share_out=share_out,
      ulinear=ulinear,
      cacheable=cacheable,
      share_layers=share_layers,
      share_mid_mul=share_mid_mul,
      mmlp=mmlp,
      ignore_purpose=ignore_purpose,
      **ignore)

    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab
    self.max_len = max_len
    self.tok_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)

    if ulinear:
      linear = ULinear
    else:
      linear = nn.Linear

    if share_in == True or share_out == True:
      if mmlp == False:
        shared_net = SimpleMLP(int(embed_dim * exp_mul), embed_dim, mid_features=int(embed_dim * share_mid_mul),
                               bias=False, layers=share_layers, linear=linear)
      else:
        shared_net = MultiMLP(int(embed_dim * exp_mul), int(embed_dim * share_mid_mul), linear=linear,
                              layers=share_layers - 1, last_activation=last_activation)
      self.shared_net = shared_net

    if share_in == True:
      share_in = self.shared_net

    if share_out == True:
      share_out = self.shared_net

    # add in the DULinear to the block definition
    dulinear = partial(SDULinear,
                       exp_mul=exp_mul,
                       predict_bias=predict_bias,
                       predict_mbias=predict_mbias,
                       predict_mbias2=predict_mbias2,
                       predict_mbias_a=predict_mbias_a,
                       predict_mbias2_a=predict_mbias2_a,
                       share_in=share_in,
                       share_out=share_out,
                       linear=linear,
                       ignore_purpose=ignore_purpose,
                       cacheable=cacheable)
    dulinear = accepts_purpose(dulinear)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        linear=dulinear,
                                        ln_attn=ln_attn,
                                        ln_mlp=ln_mlp) for _ in range(num_blocks)])
    if ln_fc:
      self.ln = nn.LayerNorm(embed_dim)
    else:
      self.ln = lambda x: x
    if dl_fc == True:
      # The out features are massive. Predicting things associated with them is too expensive.
      # If prediction was turned on we will instead set it to 'False' to use a standard ULinear which is way cheaper.
      pmb2 = predict_mbias2
      if pmb2 == True:
        pmb2 = False
      pmb2a = predict_mbias2_a
      if pmb2a == True:
        pmb2a = False
      pb = predict_bias
      if pb == True:
        pb = False

      self.fc = SDULinear(embed_dim,
                          vocab_size,
                          exp_mul=exp_mul,
                          predict_bias=pb,
                          predict_mbias=predict_mbias,
                          predict_mbias2=pmb2,
                          predict_mbias_a=predict_mbias_a,
                          predict_mbias2_a=pmb2a,
                          share_in=share_in,
                          share_out=share_out,
                          linear=linear,
                          purpose="fc",
                          ignore_purpose=ignore_purpose,
                          cacheable=False)
    elif dl_fc == False:
      self.fc = ULinear(embed_dim, vocab_size)
    else:
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
