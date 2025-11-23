from .common import Component
from .common_runner import LMRunnerCommon
from lmplay.train.datasets.plan import get_step_names
from lmplay.train.datasets.plan_configs import DEFAULT_PLANS
import json, os


class CurriculumComponent(Component):
  def __init__(self, mr: LMRunnerCommon):
    super().__init__()
    self.mr = mr

  def archive(self) -> dict:
    """Archive curriculum state.

    Saves the training plan name/dict to state and current_step to state_args.

    Returns:
      dict with 'state_args' (current_step) and 'state' (training plan)
    """
    return {
            'state_args': {
                    'current_step': self.mr.current_step,
            },
            'state': {
                    'training_plan_name': self.mr.training_plan_name,
                    'training_plan': self.mr.training_plan,
            }
    }

  def advertise(self) -> dict:
    """Advertise curriculum construction arguments.

    Exposes training_plan (filename/name of plan to load) and override_plan flag.

    Returns:
      dict with training plan configuration
    """
    return {
            'training_plan': 'default',
            # Name or path of training plan (e.g., 'default', 'full', or '/path/to/plan.json')
            'override_plan': False,  # Set to True to load a new plan and discard saved state
    }

  def construct(self, construction_args: dict, state_args: dict, state: dict):
    """Construct curriculum by loading training plan and setting current step.

    Loads the training plan from construction_args['training_plan'] if:
    - override_plan=True, or
    - No checkpoint state exists

    Sets current_step to the first step of the training plan, then overrides
    with state_args['current_step'] if provided (allows manual step override).

    Args:
      construction_args: Curriculum construction arguments (training_plan name, override_plan flag)
      state_args: Saved curriculum state arguments (current_step to override)
      state: Saved curriculum state (training_plan_name and training_plan from checkpoint)
    """
    if not self.mr.for_train:
      #Nothing to construct if we aren't training
      return

    override_plan = construction_args.get('override_plan', False) if construction_args else False
    training_plan_name = construction_args.get('training_plan') if construction_args else 'default'

    # Determine if we should load fresh from construction_args or use checkpoint state
    should_load_fresh = override_plan or not state or 'training_plan' not in state

    if should_load_fresh and training_plan_name:
      # Load training plan from file or DEFAULT_PLANS
      if training_plan_name in DEFAULT_PLANS:
        self.mr.training_plan = DEFAULT_PLANS[training_plan_name]
      else:
        # Try to load from file
        with open(os.path.expanduser(training_plan_name)) as infile:
          self.mr.training_plan = json.loads(infile.read())
      self.mr.training_plan_name = training_plan_name
    else:
      # Use checkpoint state
      self.mr.training_plan_name = state.get('training_plan_name') if state else None
      self.mr.training_plan = state.get('training_plan', {}) if state else {}

    # Set current_step: start with first step from plan, then override if provided
    first_step = get_step_names(self.mr.training_plan)[0] if self.mr.training_plan else None
    self.mr.current_step = first_step

    # Override with saved current_step if provided in state_args
    if state_args and 'current_step' in state_args:
      self.mr.current_step = state_args['current_step']
