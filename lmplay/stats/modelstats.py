"""
Real-time training statistics collection and management.

This module provides the ModelStats class for collecting, storing, and
analyzing training and validation metrics during model training. It includes
automatic CSV file generation, moving average calculation, and integration
with the plotting system.

Key features:
- Real-time statistics collection during training
- Automatic CSV file writing and updating
- Moving average estimation for smoothing noisy metrics
- Separate tracking of training and validation statistics
- Token and sample counting for accurate progress tracking
- Integration with plotting utilities for visualization

The ModelStats class is designed to be lightweight and efficient, with
configurable history preservation to manage memory usage during long
training runs.
"""

from .plot import plot, find_stat_files
from .utils import SimpleEstimator
import os

class ModelStats:
  """
  Real-time training statistics collection and management.
  
  This class provides comprehensive tracking of training and validation metrics
  during model training, with automatic CSV file generation, moving average
  calculation, and efficient memory management.
  
  The class maintains separate histories for training and validation metrics,
  automatically writes statistics to CSV files at configurable intervals,
  and provides moving average estimates for smoother metric tracking.
  
  Attributes:
    basedir (str): Directory for saving statistics files
    train_filename (str): Full path to training statistics CSV file
    validate_filename (str): Full path to validation statistics CSV file
    train_history (list): Complete training history as (iter, accuracy, loss, tokens) tuples
    validate_history (list): Complete validation history as (iter, accuracy, loss, tokens) tuples
    total_train_samples (int): Total number of training samples processed
    total_validate_samples (int): Total number of validation samples processed
    total_train_tokens (int): Total number of training tokens processed
    total_validate_tokens (int): Total number of validation tokens processed
  """
  
  def __init__(self,
               basedir="./out",
               model_name="model",
               train_history=None,
               validate_history=None,
               preserve_train_history=100,
               preserve_validate_history=100,
               total_train_samples=None,
               total_validate_samples=None,
               total_validate_tokens=0,
               total_train_tokens=0,
               **other):
    """
    Initialize ModelStats for tracking training and validation metrics.
    
    Args:
      basedir (str): Directory to save statistics files. Defaults to "./out"
      model_name (str): Base name for statistics files. Defaults to "model"
      train_history (list, optional): Existing training history to restore from
      validate_history (list, optional): Existing validation history to restore from
      preserve_train_history (int): Number of recent training samples to keep in memory for averages
      preserve_validate_history (int): Number of recent validation samples to keep in memory for averages
      total_train_samples (int, optional): Total training samples count (inferred if None)
      total_validate_samples (int, optional): Total validation samples count (inferred if None)
      total_validate_tokens (int): Total validation tokens processed
      total_train_tokens (int): Total training tokens processed
      **other: Additional metadata to store
    """
    train_filename = f"{model_name}_train_stats"
    validate_filename = f"{model_name}_validate_stats"

    os.makedirs(basedir, exist_ok=True)
    self.basedir = basedir
    self.train_filename = os.path.expanduser(os.path.join(basedir, f"{train_filename}.csv"))
    self.validate_filename = os.path.expanduser(os.path.join(basedir, f"{validate_filename}.csv"))
    self.train_plot_filename = os.path.expanduser(os.path.join(basedir, f"{train_filename}.jpg"))
    self.validate_plot_filename = os.path.expanduser(os.path.join(basedir, f"{validate_filename}.jpg"))
    self.preserve_train_history = preserve_train_history
    self.preserve_validate_history = preserve_validate_history
    self.total_train_tokens = total_train_tokens
    self.total_validate_tokens = total_validate_tokens
    if validate_history is None:
      validate_history = []
    if train_history is None:
      train_history = []
    self.train_history = train_history
    self.validate_history = validate_history
    self.other = other.get('other', dict())

    self._train_accuracy = SimpleEstimator(max_history=preserve_train_history)
    self._train_loss = SimpleEstimator(max_history=preserve_train_history)
    self._validate_accuracy = SimpleEstimator(max_history=preserve_validate_history)
    self._validate_loss = SimpleEstimator(max_history=preserve_validate_history)

    if len(self.train_history) > 0:
      if total_train_samples is None:
        total_train_samples = self.train_history[-1][0]
      self.total_train_samples = total_train_samples
      for _, accuracy, loss, total_tokens in self.train_history[-self._train_accuracy.max_history:]:
        self._train_accuracy.update(accuracy)
        self._train_loss.update(loss)
    else:
      self.total_train_samples = 0

    if len(self.validate_history) > 0:
      if total_validate_samples is None:
        total_validate_samples = len(self.validate_history)
      self.total_validate_samples = total_validate_samples
      for _, accuracy, loss, total_tokens in self.validate_history[-self._validate_accuracy.max_history:]:
        self._validate_accuracy.update(accuracy)
        self._validate_loss.update(loss)
    else:
      self.total_validate_samples = 0

    self.last_train_update = self.total_train_samples - self.total_train_samples % self.preserve_train_history
    self.last_validate_update = self.total_validate_samples - self.total_validate_samples % self.preserve_validate_history
    self.last_train_write = 0
    self.last_validate_write = 0

  def train_accuracy(self, max_history=None):
    """
    Get the current moving average of training accuracy.
    
    Args:
      max_history (int, optional): Maximum number of recent samples to include in average.
                                   If None, uses all available history.
    
    Returns:
      float or None: Moving average of training accuracy, or None if no data available.
    """
    return self._train_accuracy.estimate(max_history=max_history)

  def validate_accuracy(self, max_history=None):
    """
    Get the current moving average of validation accuracy.
    
    Args:
      max_history (int, optional): Maximum number of recent samples to include in average.
                                   If None, uses all available history.
    
    Returns:
      float or None: Moving average of validation accuracy, or None if no data available.
    """
    return self._validate_accuracy.estimate(max_history=max_history)

  def train_loss(self, max_history=None):
    """
    Get the current moving average of training loss.
    
    Args:
      max_history (int, optional): Maximum number of recent samples to include in average.
                                   If None, uses all available history.
    
    Returns:
      float or None: Moving average of training loss, or None if no data available.
    """
    return self._train_loss.estimate(max_history=max_history)

  def validate_loss(self, max_history=None):
    """
    Get the current moving average of validation loss.
    
    Args:
      max_history (int, optional): Maximum number of recent samples to include in average.
                                   If None, uses all available history.
    
    Returns:
      float or None: Moving average of validation loss, or None if no data available.
    """
    return self._validate_loss.estimate(max_history=max_history)

  def update_train(self, tokens, samples, accuracy, loss, actual_samples:int=None):
    """
    Update training statistics with new metrics.
    
    Adds new training metrics to the history, updates moving averages,
    and automatically writes to CSV file when sufficient data has accumulated.
    
    Args:
      tokens (int): Number of tokens processed in this batch
      samples (int): Number of samples processed in this batch 
      accuracy (float): Training accuracy for this batch
      loss (float): Training loss for this batch
      actual_samples (int, optional): Actual sample count if different from samples
    """
    self._train_accuracy.update(accuracy)
    self._train_loss.update(loss)
    self.total_train_tokens += tokens
    if actual_samples:
      self.total_train_samples += actual_samples
    else:
      self.total_train_samples += samples
    self.train_history.append((self.total_train_samples, accuracy, loss, self.total_train_tokens))
    if self.total_train_samples - self.last_train_update >=  self.preserve_train_history:
      #max_history = max(self.total_train_samples - self.last_train_update, 1)
      #self.train_history.append((self.total_train_samples, self.train_accuracy(max_history=max_history), self.train_loss(max_history=max_history)))
      self.write_train()
      #self._plot(self.train_filename, self.train_plot_filename)
      self.last_train_update = self.total_train_samples - self.total_train_samples % self.preserve_train_history

  def update_validate(self, tokens, samples, accuracy, loss, actual_samples:int=None):
    """
    Update validation statistics with new metrics.
    
    Adds new validation metrics to the history, updates moving averages,
    and automatically writes to CSV file when sufficient data has accumulated.
    
    Args:
      tokens (int): Number of tokens processed in this batch
      samples (int): Number of samples processed in this batch
      accuracy (float): Validation accuracy for this batch
      loss (float): Validation loss for this batch
      actual_samples (int, optional): Actual sample count if different from samples
    """
    self.total_validate_tokens += tokens
    self._validate_accuracy.update(accuracy)
    self._validate_loss.update(loss)
    if actual_samples:
      self.total_validate_samples += actual_samples
    else:
      self.total_validate_samples += samples
    self.validate_history.append((self.total_train_samples, accuracy, loss, self.total_train_tokens))
    if self.total_validate_samples - self.last_validate_update >= self.preserve_validate_history:
      #max_history = max(self.total_validate_samples - self.last_validate_update, 1)
      #self.validate_history.append((self.total_train_samples, self.validate_accuracy(max_history=max_history), self.validate_loss(max_history=max_history)))
      self.write_validate()
      #self._plot(self.validate_filename, self.validate_plot_filename)
      self.last_validate_update = self.total_validate_samples - self.last_validate_update % self.preserve_validate_history

  def _write_stats(self, location, stats, last_write):
    """
    Internal method to write statistics to CSV file.
    
    Handles both initial file creation (with headers) and incremental
    appending of new statistics data.
    
    Args:
      location (str): File path to write statistics
      stats (list): List of (iter, accuracy, loss, tokens) tuples
      last_write (int): Index of last written entry for incremental updates
    """
    if last_write == 0:
      #write whatever we have. This forces an over-write of the existing file.
      with open(location, mode="w+") as out_file:
        out_file.write("iter,accuracy,loss,tokens\n")
        for iter, accuracy, loss, tokens in stats:
          out_file.write(f"{iter},{accuracy},{loss},{tokens}\n")
    else:
      #Just append since this isn't our first write.
      with open(location, mode="a") as out_file:
        #out_file.write("iter,accuracy,loss\n")
        for iter, accuracy, loss, tokens in stats[last_write:]:
          out_file.write(f"{iter},{accuracy},{loss},{tokens}\n")

  def write_train(self, location: str = None):
    """
    Write training statistics to CSV file.
    
    Args:
      location (str, optional): File path to write to. If None, uses default train filename.
    """
    if location is None:
      location = self.train_filename
    self._write_stats(location, self.train_history, self.last_train_write)
    self.last_train_write = len(self.train_history)

  def write_validate(self, location: str = None):
    """
    Write validation statistics to CSV file.
    
    Args:
      location (str, optional): File path to write to. If None, uses default validate filename.
    """
    if location is None:
      location = self.validate_filename
    self._write_stats(location, self.validate_history, self.last_validate_write)
    self.last_validate_write = len(self.validate_history)

  def dump_dict(self) -> dict:
    """
    Export all statistics data as a dictionary for serialization.
    
    Returns:
      dict: Complete statistics data including histories, counts, and metadata.
            Contains keys: validate_history, train_history, total_validate_samples,
            total_train_samples, total_train_tokens, total_validate_tokens, other.
    """
    return {'validate_history': self.validate_history,
            'train_history': self.train_history,
            'total_validate_samples':self.total_validate_samples,
            'total_train_samples':self.total_train_samples,
            'total_train_tokens':self.total_train_tokens,
            'total_validate_tokens':self.total_validate_tokens,
            'other': self.other}

  def _plot(self, stats_file, plot_file):
    """
    Internal method to generate plots comparing current model to baselines.
    
    Args:
      stats_file (str): Path to current model's statistics CSV file
      plot_file (str): Output path for generated plot image
    """
    _, baseline_stats_files = find_stat_files(self.basedir)
    if stats_file not in baseline_stats_files:
      stats_files = baseline_stats_files + (stats_file,)
    else:
      stats_files = baseline_stats_files
    plot(plot_file, *stats_files)
