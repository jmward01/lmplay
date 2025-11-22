"""
Utility functions and classes for statistics processing and analysis.

This module provides helper classes and functions for statistical analysis
of training data, including moving average estimation and text comparison
metrics for accuracy calculation.

Key components:
- SimpleEstimator: Efficient moving average calculation with bounded history
- levenshtein_edit_distance: Text comparison for accuracy metrics

The utilities are designed to be lightweight and efficient for real-time
use during training while providing robust statistical estimates.
"""

from Levenshtein import editops

class SimpleEstimator:
  """
  Efficient moving average estimator with bounded history.
  
  Maintains a rolling window of recent values and provides fast
  moving average calculation. Automatically manages memory by
  limiting history size to prevent unbounded growth during long
  training runs.
  
  This estimator is designed for real-time use during training
  where you need smooth estimates of noisy metrics like loss
  and accuracy without storing the entire training history.
  
  Attributes:
    history (list): Recent values within the history window
    max_history (int): Maximum number of values to retain
  """
  
  def __init__(self, max_history=30):
    """
    Initialize the moving average estimator.
    
    Args:
      max_history (int): Maximum number of recent values to keep.
                         Older values are automatically discarded.
                         Defaults to 30.
    """
    self.history = []
    self.max_history = max_history

  def estimate(self, max_history=None):
    """
    Calculate the moving average of recent values.
    
    Args:
      max_history (int, optional): Override the number of recent values
                                   to include in the average. If None,
                                   uses all available history.
    
    Returns:
      float or None: Moving average of the specified number of recent
                     values, or None if no values have been recorded.
    """
    if len(self.history) == 0:
      return None
    if max_history is None:
      max_history = len(self.history)
    history = self.history[-max_history:]
    return sum(history)/len(history)

  def update(self, value):
    """
    Add a new value to the history and maintain the size limit.
    
    Args:
      value (float): New value to add to the moving average calculation.
                     The oldest value will be discarded if the history
                     exceeds max_history.
    """
    self.history.append(value)
    if len(self.history) > self.max_history:
      self.history = self.history[1:]

def levenshtein_edit_distance(prediction, truth) -> (int, int):
  """
  Calculate edit distance and character matches for text comparison.
  
  Computes the Levenshtein edit distance between predicted and true text,
  returning both the number of errors (edits needed) and the number of
  matching characters. This is useful for calculating text generation
  accuracy metrics.
  
  The function processes edit operations (insertions, deletions, substitutions)
  to determine how many characters match between the prediction and ground truth.
  
  Args:
    prediction (str): Predicted text string
    truth (str): Ground truth text string
  
  Returns:
    tuple: (errors, matches) where:
           - errors (int): Number of edit operations needed to transform
                          prediction into truth
           - matches (int): Number of characters that match between the strings
  
  Example:
    >>> levenshtein_edit_distance("hello", "helo")
    (1, 4)  # 1 deletion needed, 4 characters match
  """
  oc = editops(prediction, truth)
  last = -1
  matches = 0
  errors = 0
  for code in oc:
    matches += code[1] - (last + 1)
    last = code[1]
    if code[0] == 'equal':
      matches += 1
    else:
      errors += 1
      if code[0] == 'insert':
        last = code[1] - 1
      else:
        last = code[1]
  matches += len(prediction) - (last + 1)
  return errors, matches
