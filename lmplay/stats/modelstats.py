from .plot import plot, find_stat_files
from .utils import SimpleEstimator
import os

class ModelStats:
  def __init__(self,
               basedir="./out",
               model_name="model",
               train_history=None,
               validate_history=None,
               preserve_train_history=100,
               preserve_validate_history=100,
               total_train_samples=None,
               total_validate_samples=None,
               **other):
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
      for _, accuracy, loss in self.train_history[-self._train_accuracy.max_history:]:
        self._train_accuracy.update(accuracy)
        self._train_loss.update(loss)
    else:
      self.total_train_samples = 0

    if len(self.validate_history) > 0:
      if total_validate_samples is None:
        total_validate_samples = len(self.validate_history)
      self.total_validate_samples = total_validate_samples
      for _, accuracy, loss in self.validate_history[-self._validate_accuracy.max_history:]:
        self._validate_accuracy.update(accuracy)
        self._validate_loss.update(loss)
    else:
      self.total_validate_samples = 0

    self.last_train_update = self.total_train_samples - self.total_train_samples % self.preserve_train_history
    self.last_validate_update = self.total_validate_samples - self.total_validate_samples % self.preserve_validate_history
    self.last_train_write = 0
    self.last_validate_write = 0

  def train_accuracy(self, max_history=None):
    return self._train_accuracy.estimate(max_history=max_history)

  def validate_accuracy(self, max_history=None):
    return self._validate_accuracy.estimate(max_history=max_history)

  def train_loss(self, max_history=None):
    return self._train_loss.estimate(max_history=max_history)

  def validate_loss(self, max_history=None):
    return self._validate_loss.estimate(max_history=max_history)

  def update_train(self, samples, accuracy, loss, actual_samples:int=None):
    self._train_accuracy.update(accuracy)
    self._train_loss.update(loss)
    if actual_samples:
      self.total_train_samples += actual_samples
    else:
      self.total_train_samples += samples
    self.train_history.append((self.total_train_samples, accuracy, loss))
    if self.total_train_samples - self.last_train_update >=  self.preserve_train_history:
      #max_history = max(self.total_train_samples - self.last_train_update, 1)
      #self.train_history.append((self.total_train_samples, self.train_accuracy(max_history=max_history), self.train_loss(max_history=max_history)))
      self.write_train()
      #self._plot(self.train_filename, self.train_plot_filename)
      self.last_train_update = self.total_train_samples - self.total_train_samples % self.preserve_train_history

  def update_validate(self, samples, accuracy, loss, actual_samples:int=None):
    self._validate_accuracy.update(accuracy)
    self._validate_loss.update(loss)
    if actual_samples:
      self.total_validate_samples += actual_samples
    else:
      self.total_validate_samples += samples
    self.validate_history.append((self.total_train_samples, accuracy, loss))
    if self.total_validate_samples - self.last_validate_update >= self.preserve_validate_history:
      #max_history = max(self.total_validate_samples - self.last_validate_update, 1)
      #self.validate_history.append((self.total_train_samples, self.validate_accuracy(max_history=max_history), self.validate_loss(max_history=max_history)))
      self.write_validate()
      #self._plot(self.validate_filename, self.validate_plot_filename)
      self.last_validate_update = self.total_validate_samples - self.last_validate_update % self.preserve_validate_history

  def _write_stats(self, location, stats, last_write):
    if last_write == 0:
      #write whatever we have. This forces an over-write of the existing file.
      with open(location, mode="w+") as out_file:
        out_file.write("iter,accuracy,loss\n")
        for iter, accuracy, loss in stats:
          out_file.write(f"{iter},{accuracy},{loss}\n")
    else:
      #Just append since this isn't our first write.
      with open(location, mode="a") as out_file:
        #out_file.write("iter,accuracy,loss\n")
        for iter, accuracy, loss in stats[last_write:]:
          out_file.write(f"{iter},{accuracy},{loss}\n")

  def write_train(self, location: str = None):
    if location is None:
      location = self.train_filename
    self._write_stats(location, self.train_history, self.last_train_write)
    self.last_train_write = len(self.train_history)

  def write_validate(self, location: str = None):
    if location is None:
      location = self.validate_filename
    self._write_stats(location, self.validate_history, self.last_validate_write)
    self.last_validate_write = len(self.validate_history)

  def dump_dict(self) -> dict:
    return {'validate_history': self.validate_history,
            'train_history': self.train_history,
            'total_validate_samples':self.total_validate_samples,
            'total_train_samples':self.total_train_samples,
            'other': self.other}

  def _plot(self, stats_file, plot_file):
    _, baseline_stats_files = find_stat_files(self.basedir)
    if stats_file not in baseline_stats_files:
      stats_files = baseline_stats_files + (stats_file,)
    else:
      stats_files = baseline_stats_files
    plot(plot_file, *stats_files)
