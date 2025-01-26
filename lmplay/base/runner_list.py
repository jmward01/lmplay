


MODEL_RUNNERS = dict()


def expose_runner(cli_name:str, description=None, long_name=None):
  global MODEL_RUNNERS
  if long_name is None:
    long_name = cli_name
  if description is None:
    description = f"Runs the {long_name} model."
  def w(f):
    MODEL_RUNNERS[cli_name] = {'long_name':long_name, 'description':description, 'runner':f}
    return f
  return w