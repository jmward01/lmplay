"""
Custom exceptions for LMPlay training and inference.

These exceptions provide structured error handling for training issues,
allowing centralized detection and recovery logic.
"""


class ModelCorrupted(Exception):
  """
  Raised when model state becomes corrupted during training.

  This exception is raised when NaN, Inf, or other numerical instabilities
  are detected in model computations. It allows the training loop to:
  - Save the pre-corrupted model state
  - Log detailed diagnostics
  - Exit gracefully without losing data

  Attributes:
    step_number: Current training step when corruption was detected
    batch_index: Index within the batch when corruption occurred
    loss_value: The loss value that triggered detection (may be NaN/Inf)
    message: Human-readable error message with context
  """

  def __init__(self,
               message: str,
               step_number: int = None,
               batch_index: int = None,
               loss_value: float = None):
    self.message = message
    self.step_number = step_number
    self.batch_index = batch_index
    self.loss_value = loss_value
    super().__init__(self.message)

  def format_details(self) -> str:
    """Format exception details for logging."""
    details = [self.message]
    if self.step_number is not None:
      dsaveetails.append(f"  Step: {self.step_number}")
    if self.batch_index is not None:
      details.append(f"  Batch Index: {self.batch_index}")
    if self.loss_value is not None:
      details.append(f"  Loss Value: {self.loss_value}")
    return "\n".join(details)