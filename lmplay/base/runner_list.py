"""Model runner registration system for the lmplay framework.

This module provides a decorator-based registration system for exposing
model runners to the CLI. When experiments define runners and decorate
them with @expose_runner, they automatically become available through
the lmp_trainer command.

The registration system allows experiments to:
- Define a short CLI name for the model
- Provide a description for help text
- Optionally specify a longer display name

Example:
    @expose_runner('mymodel', description='My custom model')
    def MyModelRunner():
        return BasicModelRunner(MyModel)
"""

MODEL_RUNNERS = dict()


def expose_runner(cli_name:str, description=None, long_name=None):
  """Decorator to register a model runner for CLI access.
  
  This decorator adds a model runner function to the global registry,
  making it available through the lmp_trainer --exp command.
  
  Args:
    cli_name: Short name used in CLI (e.g., 'gpt2ish')
    description: Optional description for help text
    long_name: Optional longer display name (defaults to cli_name)
    
  Returns:
    Decorator function that registers the runner
    
  Example:
    @expose_runner('myexp', description='My experiment')
    def get_runner():
        return BasicModelRunner(MyModel, max_batch_size=32)
  """
  global MODEL_RUNNERS
  if long_name is None:
    long_name = cli_name
  if description is None:
    description = f"Runs the {long_name} model."
  def w(f):
    MODEL_RUNNERS[cli_name] = {'long_name':long_name, 'description':description, 'runner':f}
    return f
  return w