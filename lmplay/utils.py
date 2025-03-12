import torch

__all__ = ['gen_mask', 'create_linear', 'accepts_purpose', 'set_accepts_purpose', 'to_name', 'pstr']


def pstr(v) -> str:
  #Hacky way to convert parameter values to name friendly strings
  if v is None:
    return 'N'
  if isinstance(v, str):
    return v
  if isinstance(v, float):
    return f"{v:0.1f}"
  return str(int(v))

def to_name(version:str, *args, **kwargs):
  name = version
  if len(args) > 0:
    name = f"{name}_{''.join(pstr(v) for v in args)}"
  if len(kwargs) > 0:
    name = f"{name}_{'_'.join(pstr(v) for v in kwargs.values())}"
  return name


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


