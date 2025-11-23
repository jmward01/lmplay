from .runner.common import Component
from .runner.common_runner import LMRunnerCommon
from .runner.optimizers import OptimizersComponent
from .runner.runner import RunnerComponent
from .runner.model import ModelComponent
from .runner.lr_schedulers import LRSchedulersComponent
from .runner.curriculum import CurriculumComponent

class LMRunnerBase(LMRunnerCommon):
  """Abstract base class for model runners that handle training and inference.

  Runners wrap models and provide high-level functionality for:
  - Model initialization and device management
  - Training loops with gradient accumulation
  - Validation and generation
  - Checkpoint saving/loading
  - Statistics tracking
  - Optimizer and scheduler management

  Attributes:
    mini_batch_size: Maximum batch size for gradient accumulation
    stats_dir: Directory for saving statistics files
  """

  def __init__(self, mini_batch_size: int = 1, stats_dir="./out_gpt"):
    super().__init__(mini_batch_size, stats_dir)

  def _add_components(self, existing_components:dict[str, Component]):
    #These will construct in order. The order does matter.
    # We can control it manually if needed too but this is simple and clear.
    existing_components['curriculum'] = CurriculumComponent(self)
    #model before runner because the model name is needed for stat construction
    existing_components['model'] = ModelComponent(self)
    existing_components['runner'] = RunnerComponent(self)
    existing_components['optimizers'] = OptimizersComponent(self)
    existing_components['lr_schedulers'] = LRSchedulersComponent(self)
