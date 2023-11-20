from Levenshtein import editops

class SimpleEstimator:
  def __init__(self, max_history=30):
    self.history = []
    self.max_history = max_history

  def estimate(self, max_history=None):
    if len(self.history) == 0:
      return None
    if max_history is None:
      max_history = len(self.history)
    history = self.history[-max_history:]
    return sum(history)/len(history)

  def update(self, value):
    self.history.append(value)
    if len(self.history) > self.max_history:
      self.history = self.history[1:]

def levenshtein_edit_distance(prediction, truth) -> (int, int):
  oc = editops(prediction, truth)
  last = -1
  matches = 0
  errors = 0
  for code in oc:
    matches += code[1] - (last + 1)
    last =code[1]
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
