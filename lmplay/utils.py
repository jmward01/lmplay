import torch

__all__ = ['gen_mask', 'create_linear', 'accepts_purpose', 'set_accepts_purpose']


def gen_mask(max_len: int) -> torch.Tensor:
  return torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)

def accepts_purpose(o) -> bool:
  if not hasattr(o, 'accepts_purpose'):
    return False
  return o.accepts_purpose

def set_accepts_purpose(o, v: bool = True):
  if not hasattr(o, 'accepts_purpose'):
    setattr(o, 'accepts_purpose', v)
  return o

def create_linear(linear_class, purpose:str, *args, **kwargs):
  #When we construct things we give them a purpose so

  if accepts_purpose(linear_class):
    l = linear_class(*args, purpose=purpose, **kwargs)
  else:
    l = linear_class(*args, **kwargs)
  return l


