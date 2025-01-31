def create_linear(linear_class, purpose:str, *args, **kwargs):
  #When we construct things we give them a purpose so

  if hasattr(linear_class, 'accepts_purpose') and linear_class.accepts_purpose == True:
    l = linear_class(*args, purpose=purpose, **kwargs)
  else:
    l = linear_class(*args, **kwargs)
  return l
