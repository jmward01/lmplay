import torch
from torch import nn
from typing import Optional, Any

from lmplay.modules import Block, ULinear, MultiMLP, SDULinear
from lmplay.utils import set_accepts_purpose
import tiktoken
from lmplay.base.base_model import LMBase
from lmplay.modules import UnifiedEmbedding
from functools import partial


def _p(v) -> str:
  if v is None:
    return 'N'
  if isinstance(v, str):
    return v
  if isinstance(v, float):
    return f"{v:0.1f}"
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
               front_embed_mul=16.0,
               mid_mul=1.0,
               exp_mul=64.0,
               for_train=True,
               keep_embed_on_cpu=False,
               ln_attn=False,  # UW get a big boost from this and it is fewer parameter/computation so not cheating!
               ln_mlp=False,  # UW get a big boost from this and it is fewer parameter/computation so not cheating!
               ue_sduw=True,
               t_sduw=True,
               ignore_purpose=False,
               dl_fc=False,
               max_predict=True,
               version="all1",
               lradd_floor=None,
               lradd_ceil=None,
               lradd_predict=None,
               lradd_simple=True,
               imbias=False,
               iambias=False,
               ambias=False,
               mulinear=False,
               norm_v=False,
               **ignore):
    # Second in the 'sacrificial' line of experiments. These models combine all the sacrificial experiments, experiments that train with extra parameters that are removed for prod.
    # This model could be re-saved after training back to a 'standard' version compatible with the gpt2ish baseline weights.
    # This specific version combines the changes from unified embeddings 1.3 (sort of) and unified weights 2.1
    name = f"{version}_{_p(ln_attn)}{_p(lradd_predict)}{_p(lradd_simple)}{_p(imbias)}{_p(iambias)}{_p(ambias)}{_p(mulinear)}{_p(norm_v)}_{_p(lradd_floor)}_{_p(lradd_ceil)}_{_p(front_embed_mul)}_{_p(mid_mul)}_{num_blocks}L_{max_len}"
    super().__init__(
      name,
      max_len=max_len,
      num_heads=num_heads,
      num_blocks=num_blocks,
      embed_dim=embed_dim,
      attn_dropout=attn_dropout,
      ff_dropout=ff_dropout,
      embed_dropout=embed_dropout,
      front_embed_mul=front_embed_mul,
      mid_mul=mid_mul,
      exp_mul=exp_mul,
      for_train=for_train,
      keep_embed_on_cpu=keep_embed_on_cpu,
      ln_attn=ln_attn,
      ln_mlp=ln_mlp,
      version=version,
      ue_sduw=ue_sduw,
      t_sduw=t_sduw,
      ignore_purpose=ignore_purpose,
      dl_fc=dl_fc,
      max_predict=max_predict,
      lradd_floor=lradd_floor,
      lradd_ceil=lradd_ceil,
      lradd_predict=lradd_predict,
      lradd_simple=lradd_simple,
      imbias=imbias,
      iambias=iambias,
      ambias=ambias,
      mulinear=mulinear,
      norm_v=norm_v,
      **ignore)
    if max_predict == True:
      max_predict_size = embed_dim * 4
    else:
      max_predict_size = None
    keep_embed_on_cpu = for_train and keep_embed_on_cpu
    self.tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = self.tokenizer.n_vocab
    expansion_size = int(exp_mul * embed_dim)
    # self.shared_net = SimpleMLP(expansion_size, embed_dim, layers=2, bias=False, linear=ULinear)
    ulinear = partial(ULinear,
                           imbias=imbias,
                           iambias=iambias,
                           ambias=ambias)
    if mulinear == True:
      mulinear = ulinear
    else:
      mulinear = nn.Linear
    self.shared_net = MultiMLP(expansion_size, embed_dim, last_activation=False, layers=0, linear=mulinear)

    self.max_len = max_len

    if t_sduw == True or ue_sduw == True:
      linear = partial(SDULinear,
                       share_in=self.shared_net,
                       share_out=self.shared_net,
                       exp_mul=exp_mul,
                       linear=ulinear,
                       ignore_purpose=ignore_purpose,
                       cacheable=True,
                       max_predict_size=max_predict_size)
      linear = set_accepts_purpose(linear)
    else:
      linear = nn.Linear
    if ue_sduw == True:
      tok_linear = linear
    elif ue_sduw == False:
      tok_linear = ulinear
    else:
      tok_linear = nn.Linear

    if t_sduw == True:
      t_linear = linear
    elif ue_sduw == False:
      t_linear = ulinear
    else:
      t_linear = nn.Linear
    integration2 = int(embed_dim*mid_mul)
    self.tok_embed = UnifiedEmbedding(vocab_size,
                                      embed_dim,
                                      front_embed_mul,
                                      integration2=integration2,
                                      keep_embed_on_cpu=keep_embed_on_cpu,
                                      linear=tok_linear)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    self.dropout = nn.Dropout(embed_dropout)
    self.blocks = nn.Sequential(*[Block(max_len,
                                        num_heads,
                                        embed_dim,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout,
                                        linear=t_linear,
                                        ln_attn=ln_attn,
                                        ln_mlp=ln_mlp,
                                        lradd=True,
                                        lradd_floor=lradd_floor,
                                        lradd_ceil=lradd_ceil,
                                        lradd_predict=lradd_predict,
                                        lradd_simple=lradd_simple,
                                        norm_v=norm_v) for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    if dl_fc == True:
      self.fc = SDULinear(embed_dim,
                          vocab_size,
                          exp_mul=exp_mul,
                          predict_bias=False,
                          predict_mbias=True,
                          predict_mbias2=False,
                          predict_mbias_a=None,
                          predict_mbias2_a=None,
                          share_in=self.shared_net,
                          share_out=self.shared_net,
                          linear=linear,
                          purpose="fc",
                          ignore_purpose=ignore_purpose,
                          cacheable=False,
                          max_predict_size=max_predict_size)
    else:
      self.fc = ulinear(embed_dim, vocab_size)

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
