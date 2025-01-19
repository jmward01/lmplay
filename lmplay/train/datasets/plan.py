from .lmpdatasets import get_dataset
from datasets import interleave_datasets, Dataset




def get_step_names(step_def: dict):
  steps = step_def['steps']
  return tuple(step.get('step_name', str(i)) for i, step in enumerate(steps))


def get_first_step_name(step_def: dict):
  return get_step_names(step_def)[0]


# A plan is simple. It defines datasets and training steps.
# Each step consists of datasets that will be randomized then interleaved and run for a given number/fraction of an epoch.
def steps(step_def: dict, save_dataset=False, current_step: str = None) -> (str, float, Dataset, Dataset):
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
        train = interleave_datasets([dataset['train'] for dataset in full_datasets])
        validation = interleave_datasets([dataset['validation'] for dataset in full_datasets])
      else:
        train = full_datasets[0]['train']
        validation = full_datasets[0]['validation']

      print(f"Loaded")
      yield step_name, step_epochs, train, validation
