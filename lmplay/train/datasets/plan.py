"""
Training plan execution for multi-stage dataset processing.

This module handles the execution of training plans, which define multi-stage
training with different datasets at each stage. A training plan consists of:

- Dataset definitions with their loading parameters
- Training steps that specify which datasets to use and for how many epochs
- Configuration for dataset interleaving and stopping strategies

The module supports:
- Sequential execution of training steps
- Dataset interleaving with configurable stopping strategies
- Step resumption for interrupted training
- Epoch-based training duration control

Training plans enable complex curricula like pretraining on general text
followed by instruction fine-tuning on conversational data.
"""

from .lmpdatasets import get_dataset
from datasets import interleave_datasets, Dataset




def get_step_names(step_def: dict):
  """
  Extract step names from a training plan definition.
  
  Args:
    step_def: Training plan dictionary containing 'steps' list
    
  Returns:
    Tuple of step names in order. Uses indices as names if step_name not specified.
    
  Example:
    plan = {'steps': [{'step_name': 'pretrain'}, {'step_name': 'finetune'}]}
    names = get_step_names(plan)  # ('pretrain', 'finetune')
  """
  steps = step_def['steps']
  return tuple(step.get('step_name', str(i)) for i, step in enumerate(steps))


def get_first_step_name(step_def: dict):
  """
  Get the name of the first training step.
  
  Args:
    step_def: Training plan dictionary
    
  Returns:
    Name of the first step
  """
  return get_step_names(step_def)[0]


def steps(step_def: dict, save_dataset=False, current_step: str = None):
  """
  Execute training steps from a training plan.
  
  A training plan defines datasets and multi-stage training steps. Each step
  specifies which datasets to use and for how many epochs. This function
  loads the datasets and yields them in sequence for training.
  
  Args:
    step_def: Training plan dictionary with 'datasets' and 'steps' keys
    save_dataset: Whether to save datasets locally during loading
    current_step: Resume from this step (skip earlier steps)
    
  Yields:
    Tuple of (step_name, epochs, train_dataset, validation_dataset)
    
  Training Plan Structure:
    {
      'datasets': {
        'dataset_name': {
          'ds_loader': 'loader_type',
          'args': [...],
          'kwargs': {...}
        }
      },
      'steps': [
        {
          'step_name': 'step1',
          'datasets': ['dataset_name1', 'dataset_name2'],
          'epochs': 1.0,
          'interleve_stopping_strategy': 'first_exhausted'  # or 'all_exhausted'
        }
      ],
      'seed': 0
    }
    
  Example:
    for step_name, epochs, train, val in steps(training_plan):
        print(f"Training {step_name} for {epochs} epochs")
        # Train model on train dataset
  """
  seed = step_def.get('seed', 0)
  found_current = current_step is None
  for i, step in enumerate(step_def['steps']):
    step_epochs = step.get('epochs', 1.0)
    step_name = step.get('step_name', str(i))
    # Fast forward past steps we have already done.
    if current_step == step_name:
      found_current = True

    if found_current:
      print(f"Loading {step_name} for {step_epochs} epochs: {', '.join(name for name in step['datasets'])}")
      full_datasets = []
      for dataset in step['datasets']:
        dataset_info = step_def['datasets'][dataset]
        full_datasets.append(get_dataset(dataset_info['ds_loader'],
                                         *dataset_info.get('args', ()),
                                         seed=seed,
                                         save_local=save_dataset,
                                         **dataset_info.get('kwargs', dict())))

      if len(full_datasets) > 1:
        #"first_exhausted", "all_exhausted"
        interleve_stopping_strategy = step.get('interleve_stopping_strategy', "first_exhausted")
        train = interleave_datasets([dataset['train'] for dataset in full_datasets], stopping_strategy=interleve_stopping_strategy)
        validation = interleave_datasets([dataset['validation'] for dataset in full_datasets], stopping_strategy=interleve_stopping_strategy)
      else:
        train = full_datasets[0]['train']
        validation = full_datasets[0]['validation']

      print(f"Loaded")
      yield step_name, step_epochs, train, validation
