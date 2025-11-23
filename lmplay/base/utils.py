import torch
from typing import Sequence

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
  """Scale logits by temperature for controlling output diversity.

  Args:
    logits: Model output logits of shape (batch, vocab_size)
    temperature: Temperature value. < 1.0 makes distribution sharper (more deterministic),
                 > 1.0 makes it softer (more random). 1.0 is unchanged.

  Returns:
    Temperature-scaled logits
  """
  if temperature == 1.0:
    return logits
  if temperature <= 0:
    raise ValueError("Temperature must be positive")
  return logits / temperature


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor,
                             penalty: float) -> torch.Tensor:
  """Apply repetition penalty to discourage repeated tokens.

  Args:
    logits: Model output logits of shape (vocab_size,)
    input_ids: Previously generated token IDs
    penalty: Penalty factor. > 1.0 discourages repetition. 1.0 is no penalty.

  Returns:
    Logits with repetition penalty applied
  """
  if penalty == 1.0:
    return logits
  if penalty <= 0:
    raise ValueError("Repetition penalty must be positive")

  # Get unique tokens that have been generated
  unique_tokens = torch.unique(input_ids)
  for token_id in unique_tokens:
    # Reduce logit by dividing (makes tokens less likely)
    logits[token_id] = logits[token_id] / penalty

  return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
  """Filter to keep only top k tokens.

  Args:
    logits: Model output logits of shape (vocab_size,)
    top_k: Number of highest probability tokens to keep. 0 or None disables.

  Returns:
    Logits with non-top-k values set to -inf
  """
  if top_k is None or top_k <= 0:
    return logits

  top_k = min(top_k, logits.size(-1))
  top_k_logits, top_k_indices = torch.topk(logits, top_k)

  # Create mask with -inf for non-top-k tokens
  logits_mask = torch.full_like(logits, float('-inf'))
  logits_mask.scatter_(0, top_k_indices, top_k_logits)

  return logits_mask


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
  """Apply nucleus sampling (top-p filtering).

  Args:
    logits: Model output logits of shape (vocab_size,)
    top_p: Cumulative probability threshold. Tokens with cumulative prob > top_p are filtered.
           1.0 disables. Typical values: 0.9, 0.95

  Returns:
    Logits with tokens outside cumulative probability set to -inf
  """
  if top_p >= 1.0:
    return logits
  if top_p <= 0:
    raise ValueError("top_p must be in (0, 1]")

  # Sort probabilities in descending order
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  sorted_probs = torch.softmax(sorted_logits, dim=-1)

  # Compute cumulative probabilities
  cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

  # Keep tokens until cumulative prob exceeds top_p, but always keep at least one token
  mask = cumsum_probs <= top_p
  mask[0] = True  # Always keep the highest probability token
  sorted_logits[~mask] = float('-inf')

  # Unsort back to original order
  result = torch.full_like(logits, float('-inf'))
  result.scatter_(0, sorted_indices, sorted_logits)

  return result


class NopWith:
  """No-op context manager for conditional context usage.

  Used as a placeholder when AMP (automatic mixed precision) is disabled,
  allowing the same code structure regardless of AMP status.
  """

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass