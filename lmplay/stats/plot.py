"""
Advanced plotting and visualization system for training statistics.

This module provides a comprehensive plotting system for visualizing training
and validation metrics from language model experiments. It includes sophisticated
features for multi-experiment comparison, data smoothing, outlier detection,
and differential analysis.

Key features:
- Multi-experiment comparison with automatic legend generation
- Advanced smoothing with adaptive averaging windows
- Outlier detection and removal for cleaner visualizations
- Differential plotting to highlight performance differences
- Log-scale plotting for better visualization of training curves
- Raw data scatter plots with smoothed trend overlays
- Automatic baseline detection and comparison
- Memory-efficient processing using multiprocessing
- Publication-quality output with configurable formats

The plotting system uses tokens as the primary x-axis metric for more
meaningful comparisons across different batch sizes and training configurations.
It supports both training and validation statistics with automatic file
discovery and intelligent data processing.

Data Processing Pipeline:
1. File discovery and CSV parsing
2. Data unification across different experiment lengths
3. Outlier detection and bounds calculation
4. Adaptive smoothing with triangular weighting
5. Optional differential analysis against baseline
6. Multi-process rendering for memory management
7. Publication-quality output generation

Constants:
  X_KEY (str): CSV column name for x-axis data (tokens)
  X_LABEL (str): Display label for x-axis (Tokens)
"""

import math
import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt

import csv
import multiprocessing as mp

import numpy
import numpy as np
import bisect

# Primary x-axis metric for plotting - using tokens provides more meaningful
# comparisons across different batch sizes and training configurations
X_KEY = "tokens"
X_LABEL = "Tokens"

def find_stat_files(path: str) -> (tuple, tuple):
  """
  Discover and categorize training statistics CSV files in a directory.
  
  Searches for files ending with '_stats.csv' and separates them into
  regular experiment files and baseline files (containing 'baseline' in name).
  Files are sorted by modification time with most recent first.
  
  Args:
    path (str): Directory path to search for statistics files
  
  Returns:
    tuple: (experiment_files, baseline_files) - both as tuples of file paths
           sorted by modification time (newest first)
  """
  path = os.path.expanduser(path)
  found_files = dict()
  found_baseline_files = dict()
  try:
    for file in os.listdir(path):
      if file.endswith('_stats.csv') and not file.startswith('._'):
        file = os.path.join(path, file)
        mtime = os.stat(file).st_mtime
        if 'baseline' in file.lower():
          found_baseline_files[file] = mtime
        else:
          found_files[file] = mtime
  except OSError:
    pass
  found_files = tuple(reversed(tuple(sorted(found_files, key=found_files.get, reverse=True))))
  found_baseline_files = tuple(
    reversed(tuple(sorted(found_baseline_files, key=found_baseline_files.get, reverse=True))))
  return found_files, found_baseline_files

def max_toss_outliers(d:np.ndarray|list[float], prev_max:float|None, worst_mul=10.0) -> float:
  """
  Calculate a robust maximum value with outlier detection and removal.
  
  Uses the 80th-90th percentile slope to extrapolate a reasonable maximum
  that excludes extreme outliers while preserving important data characteristics.
  
  Args:
    d (np.ndarray|list[float]): Data array to analyze
    prev_max (float|None): Previous maximum to ensure monotonic behavior
    worst_mul (float): Multiplier for worst-case outlier detection
  
  Returns:
    float: Robust maximum value with outliers excluded
  """
  if isinstance(d, list):
    d = np.array(d)
  d = np.sort(d)
  start_idx = int(d.shape[0]*.8)
  end_idx = int(d.shape[0]*.9)
  full_slope = (d[end_idx] - d[start_idx])/(end_idx-start_idx)
  worst_max = float(full_slope*int(d.shape[0]*.1)*worst_mul + d[end_idx])
  cleaned_max = min(worst_max, float(d[-1]))
  if not prev_max is None:
    cleaned_max = max(prev_max, cleaned_max)
  return cleaned_max


def min_toss_outliers(d:np.ndarray|list[float], prev_min:float|None, worst_mul=10.0) -> float:
  """
  Calculate a robust minimum value with outlier detection and removal.
  
  Uses the 10th-20th percentile slope to extrapolate a reasonable minimum
  that excludes extreme outliers while preserving important data characteristics.
  
  Args:
    d (np.ndarray|list[float]): Data array to analyze
    prev_min (float|None): Previous minimum to ensure monotonic behavior
    worst_mul (float): Multiplier for worst-case outlier detection
  
  Returns:
    float: Robust minimum value with outliers excluded
  """
  if isinstance(d, list):
    d = np.array(d)
  d = np.sort(d)
  start_idx = int(d.shape[0]*.1)
  end_idx = int(d.shape[0]*.2)
  full_slope = (d[start_idx] - d[end_idx])/(end_idx-start_idx)
  worst_min = float(full_slope*int(d.shape[0]*.1)*worst_mul + d[start_idx])
  cleaned_min = max(worst_min, float(d[0]))
  if not prev_min is None:
    cleaned_min = min(prev_min, cleaned_min)
  return cleaned_min


def _plot_worker(out_file,
                 file_data: dict,
                 plot_targets,
                 log_plot: bool,
                 show: bool,
                 scale: bool,
                 min_iter: int,
                 max_iter: int,
                 abs_min_value: float,
                 abs_max_value: float,
                 max_min_value: float,
                 min_max_value: float,
                 min_show: float,
                 plot_raw: bool,
                 average_count: Optional[int],
                 satart_ac=5,
                 ac_inc=6,
                 target_points=40000):
  """
  Core plotting worker function with advanced smoothing and visualization.
  
  Generates publication-quality plots with adaptive smoothing, outlier removal,
  and sophisticated data processing. Uses triangular weighting for smooth curves
  and handles both raw scatter plots and trend lines.
  
  Args:
    out_file (str): Output file path for the generated plot
    file_data (dict): Dictionary mapping experiment names to data arrays
    plot_targets (tuple): Metric names to plot (e.g., 'loss', 'accuracy')
    log_plot (bool): Use logarithmic x-axis scaling
    show (bool): Display plot in interactive window
    scale (bool): Auto-scale axes to focus on interesting regions
    min_iter (int): Minimum x-axis value to display
    max_iter (int): Maximum x-axis value to display  
    abs_min_value (float): Global minimum y-value across all data
    abs_max_value (float): Global maximum y-value across all data
    max_min_value (float): Maximum of all dataset minimums
    min_max_value (float): Minimum of all dataset maximums
    min_show (float): Fraction of shortest run to display
    plot_raw (bool): Show raw data points as scatter plot
    average_count (Optional[int]): Maximum smoothing window size
    satart_ac (int): Initial smoothing window size
    ac_inc (int): Smoothing window increment rate
    target_points (int): Target number of points for performance optimization
  """
  # with plt.xkcd():
  acs = dict()
  #This is horrible nog good bad code. Sorry. Please don't think of this as something to use.
  # It is something I tweak all the time to get things to look the way I want and not good code.
  #def get_ac(i):
  #  if i not in acs:
  #    ac = []
  #    center = int(i / 2)
  #    for j in range(i * 2 + 1):
  #      weight = (1.0 - abs(j - center) / i) ** 2
  #      ac.append(weight)
  #    ac = numpy.array(ac)
  #    acs[i] = ac
  #  return acs[i]

  def get_ac(i, center):
    """
    Generate triangular averaging weights for smooth data interpolation.
    
    Creates normalized triangular weights centered around a specific point
    for sophisticated data smoothing that preserves trends while reducing noise.
    
    Args:
      i (int): Window size for averaging
      center (int): Center position within the window
    
    Returns:
      np.ndarray: Normalized triangular weights summing to 1.0
    """
    if (i,center) not in acs:
      ac = []
      #center = int(i / 2)
      for j in range(i):
        weight = (1.0 - abs(j - center) / i) ** 2
        ac.append(weight)
      ac = numpy.array(ac)
      ac = ac / np.sum(ac)
      acs[(i,center)] = ac
    return acs[(i,center)]


  # ac = ac/np.sum(ac)
  abs_avg_min_value = None
  abs_avg_max_value = None
  abs_min_value = None
  abs_max_value = None
  addtl_lines = []
  for name, data in file_data.items():
    iters = data[X_KEY]
    d_stride = math.ceil(len(iters)/target_points)
    #d_stride = 1
    if d_stride > 1:
      iters = iters[::d_stride]
    first_d = 0
    #bisect just gives a value which is stupid. I want an index
    while first_d < len(iters) and iters[first_d] < min_iter:
      first_d += 1
    #first_d = 0
    for data_name in plot_targets:
      d = numpy.array(data[data_name])
      # if average_count is None:
      #  plt.plot(data[X_KEY], d, label=f"{name}_{data_name}", linewidth=1)
      # else:
      #x_axis = data[X_KEY]
      #d_stride = math.ceil(len(d)/target_points)
      if d_stride > 1:
        d =d[::d_stride]
        #x_axis = x_axis[::d_stride]
      if plot_raw:
        l = plt.plot(data[X_KEY], d, linewidth=.2, alpha=.3)
      else:
        l = None
      avgs = [0.0] * len(d)
      current_average_count = satart_ac
      # inc = max(int(len(d)/average_count), 1)
      inc = ac_inc * 4
      for _ in range(int(first_d/inc)):
        current_average_count = min(average_count, current_average_count + ac_inc)
      for i in range(first_d, len(d)):
        if (i + 1) % inc == 0:
          current_average_count = min(average_count, current_average_count + ac_inc)
        #ac = get_ac(current_average_count)
        # Start and end are centered around i
        #d_end = min(i + current_average_count, len(d))
        d_start = max(0, i - current_average_count)
        d_end = min(len(d), i + current_average_count)
        count = d_end - d_start
        ac = get_ac(count, i - d_start)
        # count = min(i - d_start, d_end - i - 1)
        # d_start = i - count
        # d_end = i + count + 1
        #ac_start = current_average_count - count
        #ac_end = current_average_count + count + 1
        d_sect = d[d_start:d_end]
        #ac_sect = ac[ac_start:ac_end]
        ac_sect = ac
        ac_sect = ac_sect / np.sum(ac_sect)
        avg = d_sect * ac_sect
        avg = np.sum(avg)
        avgs[i] = avg

      t_min = min(d[first_d:])
      t_max = max(d[first_d:])
      abs_avg_max_value = max_toss_outliers(avgs[first_d:], abs_avg_max_value)
      abs_avg_min_value = min_toss_outliers(avgs[first_d:], abs_avg_min_value)
      abs_max_value = max_toss_outliers(d[first_d:], abs_max_value)
      abs_min_value = min_toss_outliers(d[first_d:], abs_min_value)

      #if abs_avg_min_value is None:
      #  abs_avg_min_value = min(avgs[first_d:])
      #  abs_avg_max_value = max(avgs[first_d:])
      #else:
      #  abs_avg_min_value = min(min(avgs[first_d:]), abs_avg_min_value)
      #  abs_avg_max_value = max(max(avgs[first_d:]), abs_avg_max_value)
      #if abs_min_value is None:
      #  abs_min_value = float(d[first_d:].min())
      #  abs_max_value = float(d[first_d:].max())
      #else:
      #  abs_min_value = min(abs_min_value, float(d[first_d:].min()))
      #  abs_max_value = max(abs_max_value, float(d[first_d:].max()))
      if abs_max_value == abs_min_value:
        abs_max_value = None
        abs_min_value = None
      if abs_avg_min_value == abs_avg_max_value:
        abs_avg_max_value = None
        abs_avg_min_value = None

      if l is None:
        addtl_lines.append({X_KEY: iters, 'data': avgs, "label": f"{name}_{data_name}"})
      else:
        addtl_lines.append(
          {X_KEY: iters, 'data': avgs, 'color': l[0].get_color(), "label": f"{name}_{data_name}"})
  for line_info in addtl_lines:
    # do these last so the show on top of the other line data
    if plot_raw:
       add_width = 0.0
    else:
      add_width = 0.5
    if 'color' in line_info:
      plt.plot(line_info[X_KEY], line_info['data'], label=line_info['label'], linewidth=.5 + add_width, color=line_info['color'])
    else:
      plt.plot(line_info[X_KEY], line_info['data'], label=line_info['label'], linewidth=.3 + add_width)
  plt.ylabel(' '.join(plot_targets))
  plt.xlabel(X_LABEL)
  if log_plot:
    plt.xscale('log')
  if not plot_raw:
    abs_min_value = abs_avg_min_value
    abs_max_value = abs_avg_max_value

  # We want to scale it just the lowest part to zoom in on it so set the max to min_show above out max_min
  if scale:
    # upper_show = (abs_max_value - max_min_value) * min_show
    # upper_show += max_min_value
    # y_min = abs_min_value - .001 * upper_show
    # y_max = upper_show
    if abs_max_value == abs_min_value:
      if abs_min_value is None:
        abs_min_value = 0.0
        abs_max_value = 0.0
      abs_max_value += .1
    y_min = abs_min_value
    y_max = abs_max_value
    plt.ylim(y_min, y_max)
    # x_min = max(min_iter - max(min_iter * .2, 100), 1)
    x_min = max(min_iter, 1)
    plt.xlim(x_min, max_iter * 1.1)
  # plt.autoscale(enable=True, axis='both', tight=True)
  plt.grid(True)
  leg = plt.legend(bbox_to_anchor=(1.05, 1),
             loc='upper left',
             borderaxespad=0.)
  for legobj in leg.get_lines():
      legobj.set_linewidth(2.0)
  plt.savefig(out_file, bbox_inches='tight', dpi=600)
  if show:
    plt.show()
  plt.clf()
  plt.close("all")


def _plot(*args):
  """
  Memory-safe plotting wrapper using multiprocessing.
  
  Matplotlib has memory leaks during intensive plotting. This function
  creates a separate process for plotting and kills it afterwards to
  ensure memory is properly released.
  
  Args:
    *args: Arguments passed directly to _plot_worker
  """
  # matplotlib has a memory leak. This constructs a process then kills it which clears the used memory.
  proc = mp.Process(target=_plot_worker, args=args)
  proc.daemon = True
  proc.start()
  proc.join()


def get_stats(file_data: dict, plot_targets: tuple, shortest_iter: int):
  """
  Calculate global statistics across all datasets for plot scaling.
  
  Analyzes all data to determine appropriate axis bounds and scaling
  parameters for consistent visualization across multiple experiments.
  
  Args:
    file_data (dict): Dictionary mapping experiment names to data arrays
    plot_targets (tuple): Metric names to analyze
    shortest_iter (int): Minimum iteration threshold for inclusion
  
  Returns:
    tuple: (abs_min_value, abs_max_value, max_min_value, min_max_value)
           Global statistics for plot scaling
  """
  abs_min_value = None
  abs_max_value = None
  max_min_value = None
  min_max_value = None
  for name, data in file_data.items():
    dataset_min = None
    dataset_max = None
    for i, iter_val in enumerate(data[X_KEY]):
      if iter_val != 0 and iter_val >= shortest_iter:
        for data_name in plot_targets:
          value = data[data_name][i]
          if dataset_min is None:
            dataset_min = value
          if dataset_max is None:
            dataset_max = value
          dataset_min = min(dataset_min, value)
          dataset_max = max(dataset_max, value)

    # now figure out our tracked max /min values across datasets
    if abs_max_value is None:
      abs_max_value = dataset_max
    if abs_min_value is None:
      abs_min_value = dataset_min
    if max_min_value is None:
      max_min_value = dataset_min
    if min_max_value is None:
      min_max_value = dataset_max

    abs_max_value = max(dataset_max, abs_max_value)
    abs_min_value = min(dataset_min, abs_min_value)
    max_min_value = max(max_min_value, dataset_min)
    min_max_value = min(min_max_value, dataset_max)

    # upper_show = (dataset_max - dataset_min) * min_show + dataset_min

  return abs_min_value, abs_max_value, max_min_value, min_max_value


def unify_points(file_data: dict, iters: list, plot_targets: tuple) -> dict:
  """
  Unify data points across experiments with different iteration counts.
  
  Creates aligned datasets where all experiments have data points at the
  same iteration values, enabling proper comparison and differential analysis.
  
  Args:
    file_data (dict): Single experiment's data arrays
    iters (list): Master list of all iteration values across experiments
    plot_targets (tuple): Metric names to unify
  
  Returns:
    dict: Unified data with consistent iteration points
  """
  new_iters = []
  result_file = {X_KEY: new_iters}
  for pt in plot_targets:
    file_idx = 0
    file_iters = file_data[X_KEY]
    file_values = file_data[pt]
    result_values = []
    result_file[pt] = result_values
    for iter in iters:
      if file_idx < len(file_iters):
        result_values.append(file_values[file_idx])
        new_iters.append(iter)
        # since the iters passed in contain all possible iters we are either on this iter or ahead of it by some amount.
        if file_iters[file_idx] == iter:
          # looks like we were on it. Advance to the next one
          file_idx += 1
      else:
        break
  return result_file


def _get_iters(files_data: dict) -> dict:
  """
  Extract iteration metadata from multiple experiment datasets.
  
  Analyzes all experiments to find the complete set of iteration values
  and identifies the longest and shortest runs for comparison purposes.
  
  Args:
    files_data (dict): Dictionary mapping experiment names to data arrays
  
  Returns:
    dict: Contains 'iters' (sorted list), 'longest' (experiment name),
          'shortest' (experiment name)
  """
  found_iters = set()
  longest = None
  longest_len = None
  shortest = None
  shortest_len = None
  # Find all the 'iters' in every file so we can build values for all files.
  for name, data in files_data.items():
    found_iters.update(data[X_KEY])
    if longest is None or longest_len < data[X_KEY][-1]:
      longest = name
      longest_len = data[X_KEY][-1]
    if shortest is None or shortest_len > data[X_KEY][-1]:
      shortest = name
      shortest_len = data[X_KEY][-1]
  iters = list(found_iters)
  iters.sort()
  return {'iters': iters, 'longest': longest, 'shortest': shortest}


def _diff_to_target(files_data: dict, target: str, plot_targets: tuple) -> dict:
  """
  Create differential plots by subtracting target experiment values.
  
  Transforms absolute metric values into differences from a baseline,
  making it easier to see relative performance improvements or degradations.
  
  Args:
    files_data (dict): Dictionary mapping experiment names to data arrays
    target (str): Name of target/baseline experiment to subtract
    plot_targets (tuple): Metric names to transform
  
  Returns:
    dict: Transformed data with differential values
  """
  target = files_data[target]
  result_files = dict()
  # Then we subtract the longest one's value from their values to normalize against it.
  for file_name, data in files_data.items():
    result_file = {X_KEY: data[X_KEY].copy()}
    result_files[file_name] = result_file
    for pt in plot_targets:
      target_values = []
      result_file[pt] = target_values
      for i in range(len(data[X_KEY])):
        if len(target[pt]) > i:
          target_values.append(data[pt][i] - target[pt][i])
        else:
          # Looks like the target is shorter than this run. We should trim it and not display the rest.
          result_file[X_KEY] = result_file[X_KEY][:i]
          break
  return result_files


def get_file_data(*files, plot_targets=('loss', 'accuracy')) -> (dict, dict):
  """
  Load and parse CSV statistics files into plottable data structures.
  
  Reads multiple CSV files, extracts specified metrics, and creates
  a unified data structure for plotting. Handles name deduplication
  and filters out zero-iteration entries.
  
  Args:
    *files: Paths to CSV statistics files
    plot_targets (tuple): Metric column names to extract
  
  Returns:
    tuple: (file_data_dict, metadata_dict) where file_data maps experiment
           names to data arrays and metadata contains iteration information
  """
  file_data = dict()
  for file in files:
    file = os.path.expanduser(file)
    with open(file) as infile:
      file = os.path.basename(file)
      if '_stats' in file:
        basename = file.split('_stats')[0]
      elif '.csv' in file:
        basename = file.split('.csv')[0]
      else:
        basename = file
      name = basename
      name_count = 1
      while name in file_data:
        name_count += 1
        name = f"{basename}_{name_count}"
      csv_data = csv.DictReader(infile)
      data = {data_name: [] for data_name in plot_targets}
      data[X_KEY] = []
      for row in csv_data:
        iter_val = int(row[X_KEY])
        if iter_val != 0:
          for data_name in plot_targets:
            value = float(row[data_name])
            data[data_name].append(value)
          data[X_KEY].append(iter_val)
      if len(data[X_KEY]) > 0:
        file_data[name] = data
  return file_data, _get_iters(file_data)


def plot(out_file,
         file_data: (dict, dict),
         min_show=.1,
         log_plot=True,
         show=False,
         plot_targets=('loss',),
         scale=True,
         average_count: Optional[int] = 10,
         diff_to_target=False,
         use_process=True,
         target=None,
         plot_raw=False):
  """
  Main plotting function for training statistics visualization.
  
  Creates publication-quality plots comparing multiple experiments with
  advanced features like smoothing, outlier removal, differential analysis,
  and memory-safe rendering.
  
  Args:
    out_file (str): Output file path for the generated plot
    file_data (tuple): (data_dict, metadata_dict) from get_file_data()
    min_show (float): Fraction of shortest run to display (0.0-1.0)
    log_plot (bool): Use logarithmic x-axis scaling
    show (bool): Display plot in interactive window
    plot_targets (tuple): Metric names to plot (e.g., 'loss', 'accuracy')
    scale (bool): Auto-scale axes to focus on interesting regions
    average_count (Optional[int]): Maximum smoothing window size
    diff_to_target (bool): Create differential plot against baseline
    use_process (bool): Use multiprocessing for memory safety
    target (str, optional): Target experiment name for differential plots
    plot_raw (bool): Show raw data points as scatter plot
  
  The function automatically handles data unification, outlier detection,
  adaptive smoothing, and creates a professional visualization with proper
  legends, grid lines, and axis scaling.
  """
  out_file = os.path.expanduser(out_file)
  # We want to center the graph on the interesting areas so we need to track the overall min/max and the worst min value
  # across all datasets we are plotting. Then we will show from the absolute min to abve the worst min but below the abs max.

  file_data, file_meta = file_data
  # iters, longest, shortest = _get_iters(file_data)
  # now we have a spot for every iter. Let's get normalized value for every point by going through them all.
  # First we make sure that they all have values for every iter
  file_data = {name: unify_points(fd, file_meta['iters'], plot_targets) for name, fd in file_data.items()}
  min_iter = int(file_data[file_meta['shortest']][X_KEY][-1] * (1.0 - min_show))
  # min_iter = iters[0]
  max_iter = file_meta['iters'][-1]
  if diff_to_target:
    if not target is None:
      for run_name in file_data:
        if run_name.startswith(target):
          target = run_name
          break
      else:
        print(f"Couldn't find {target}. Using longest.")
        target = None
    if target is None:
      target = file_meta['longest']
    file_data = _diff_to_target(file_data, target, plot_targets)
  abs_min_value, abs_max_value, max_min_value, min_max_value = get_stats(file_data, plot_targets, min_iter)
  if len(file_data) > 0:
    if use_process:
      _p = _plot
    else:
      _p = _plot_worker
    _p(out_file,
       file_data,
       plot_targets,
       log_plot,
       show,
       scale,
       min_iter,
       max_iter,
       abs_min_value,
       abs_max_value,
       max_min_value,
       min_max_value,
       min_show,
       plot_raw,
       average_count)
