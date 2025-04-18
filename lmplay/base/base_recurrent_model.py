from .base_model import MBase
import torch
from typing import Optional, Sequence, List
from abc import abstractmethod
import torch.nn.functional as F

def _prune_cache(cache:list, keep_map):
  for entry in cache:
    for i, t in enumerate(entry):
      t = t[keep_map]
      entry[i] = t


class RMBase(MBase):
  def train_prompts(self, prompts: Sequence[dict], include_prompts=True) -> (Sequence[str], torch.Tensor):
    # We want to pad them together so that the truths will line up with the prompts.
    x, predictions_starts, predictions_ends = self._tokenize_batch(prompts)
    # Truth doesn't have the first EOT char. It needs to start on prediction start
    truths = x[:, 1:]
    cache = []
    #recurrent state starts off as None. The model can store a starting state if it wants one.
    r = None
    #While training some will finish before others. We only generate for those that are finished
    gen_map = tuple(i for i in range(x.shape[0]))
    #We don't send in the last one because that would generate one too many results
    x_out = None
    for i in range(x.shape[-1] - 1):
      remake_batch = False
      #Check to see if we can prune out a sample or not.
      keep_map = []
      new_gen_map = []
      for batch_idx, j in enumerate(gen_map):
        if predictions_ends[j] < i:
          remake_batch = True
        else:
          keep_map.append(batch_idx)
          new_gen_map.append(j)
      if remake_batch:
        _prune_cache(cache, keep_map)
        if not r is None:
          r = r[keep_map]
        gen_map = new_gen_map
      packed_xi = x[gen_map,i:i+1]
      packed_xi_out, cache, r = self(packed_xi, r, cache)
      #We keep the initial batch size and insert the result into it but we only generate for the still active samples.
      if x_out is None:
        x_out = torch.empty((x.shape[0], truths.shape[1], self.tokenizer.n_vocab), device=x.device, dtype = packed_xi_out.dtype)
      x_out[gen_map,i,:] = packed_xi_out[:,0,:]
    #From here on out should be the same as a normal LM
    results = []
    # num classes is always second. For, reasons?
    target_loss = F.cross_entropy(x_out.permute(0, 2, 1), truths, reduction="none")
    total_target_loss = 0.0
    total_token_count = 0
    for tl, prediction_start, prediction_end in zip(target_loss, predictions_starts, predictions_ends):
      if not include_prompts:
        tl = tl[prediction_start:prediction_end]
      else:
        tl = tl[:prediction_end]
        prediction_start = 0
      tl = tl.sum()
      token_count = max(prediction_end - prediction_start, 1)
      total_token_count += token_count
      # norm by number of tokens in the truth
      #total_target_loss = tl / token_count + total_target_loss
      total_target_loss = total_target_loss + tl
    #total_target_loss = total_target_loss/total_token_count
    #target_loss = total_target_loss
    # Get the predicted value so we can cut just that out as the result.
    predicted_tokens = torch.argmax(x_out, dim=-1)
    # for result, prediction_start, prediction_end, truth in zip(predicted_tokens, predictions_starts, predictions_ends, truths):
    for result, prediction_start, prediction_end in zip(predicted_tokens, predictions_starts, predictions_ends):
      # we only care about the prediction.
      # the last value is end of sentence
      results.append(self.tokenizer.decode(result[prediction_start:prediction_end].tolist()))
    # target loss is normalized by example but not by batch. That will be done by the caller.
    return results, total_target_loss, total_token_count

  def generate_prompts(self, prompts: Sequence[dict], max_len: Optional[int] == None) -> Sequence[str]:
    #BROKEN!
    results = []
    if not max_len:
      max_len = self.max_len
    with torch.no_grad():
      for prompt in prompts:
        result = []
        # We only support batch size of 1. This code is just for testing and not meant to be fast/good/etc.
        # Basically, we want training to be -really- easy to play with so we have the minor concession of a kv cache mechanism.
        x, _, _ = self._tokenize_batch([prompt])
        cache = []
        stop = False
        while not stop:
          x, cache = self(x, cache=cache)
          # Should only be one token
          x = torch.argmax(x, dim=-1)
          result.append(x.squeeze().tolist())
          if result[-1] == self.tokenizer.eot_token or len(result) == max_len:
            stop = True
        results.append(self.tokenizer.decode(result))

    return results


class RLMBase(RMBase):
  @abstractmethod
  def forward(self, x: torch.Tensor, s: torch.Tensor|None, cache: List) -> (torch.Tensor, torch.Tensor):
    pass
