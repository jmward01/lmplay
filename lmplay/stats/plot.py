import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt

import csv
import multiprocessing as mp

import numpy
import numpy as np

def find_stat_files(path: str) -> (tuple, tuple):
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


def _plot_worker(out_file,
                 file_data: dict,
                 plot_targets,
                 log_plot:bool,
                 show:bool,
                 scale: bool,
                 min_iter:int,
                 max_iter:int,
                 abs_min_value:float,
                 abs_max_value:float,
                 max_min_value:float,
                 min_max_value:float,
                 min_show:float,
                 average_count:Optional[int]):
  #with plt.xkcd():
  ac = []
  center = int(average_count/2)
  for i in range(average_count*2 + 1):
    weight = (1.0 - abs(i - center) / average_count) ** 2
    ac.append(weight)
  ac = numpy.array(ac)
  #ac = ac/np.sum(ac)
  addtl_lines = []
  for name, data in file_data.items():
    for data_name in plot_targets:
      d = numpy.array(data[data_name])
      if average_count is None:
        plt.plot(data['iter'], d, label=f"{name}_{data_name}", linewidth=1)
      else:
        l = plt.plot(data['iter'], d, linewidth=.2, alpha=.3)
        avgs = [0.0]*len(d)
        for i in range(len(d)):
          d_start = max(0, i - average_count)
          d_end = min(i + average_count, len(d))
          count = min(i - d_start, d_end - i - 1)
          d_start = i - count
          d_end = i + count + 1
          ac_start = average_count -count
          ac_end = average_count + count + 1
          d_sect = d[d_start:d_end]
          ac_sect = ac[ac_start:ac_end]
          ac_sect = ac_sect/np.sum(ac_sect)
          avg = d_sect*ac_sect
          avg = np.sum(avg)
          avgs[i] = avg
        addtl_lines.append({'iter':data['iter'], 'data':avgs, 'color':l[0].get_color(), "label": f"{name}_{data_name}"})
  for line_info in addtl_lines:
    #do these last so the show on top of the other line data
    plt.plot(line_info['iter'], line_info['data'], label=line_info['label'], linewidth=.5, color=line_info['color'])
  plt.ylabel(' '.join(plot_targets))
  plt.xlabel('Iteration')
  if log_plot:
    plt.xscale('log')
  # We want to scale it just the lowest part to zoom in on it so set the max to min_show above out max_min
  if scale:
    #upper_show = (abs_max_value - max_min_value) * min_show
    #upper_show += max_min_value
    #y_min = abs_min_value - .001 * upper_show
    #y_max = upper_show
    if abs_max_value == abs_min_value:
      abs_max_value += .1
    y_min = abs_min_value
    y_max = abs_max_value
    plt.ylim(y_min, y_max)
    #x_min = max(min_iter - max(min_iter * .2, 100), 1)
    x_min = max(min_iter, 1)
    plt.xlim(x_min, max_iter*1.1)
  # plt.autoscale(enable=True, axis='both', tight=True)
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1.05, 1),
             loc='upper left',
             borderaxespad=0.)
  plt.savefig(out_file, bbox_inches='tight', dpi=600)
  if show:
    plt.show()
  plt.clf()
  plt.close("all")


def _plot(*args):
  # matplotlib has a memory leak. This constructs a process then kills it which clears the used memory.
  proc = mp.Process(target=_plot_worker, args=args)
  proc.daemon = True
  proc.start()
  proc.join()

def get_stats(file_data:dict, plot_targets:tuple, shortest_iter:int):
  abs_min_value = None
  abs_max_value = None
  max_min_value = None
  min_max_value = None
  for name, data in file_data.items():
    dataset_min = None
    dataset_max = None
    for i, iter_val in enumerate(data['iter']):
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

    #upper_show = (dataset_max - dataset_min) * min_show + dataset_min

  return abs_min_value, abs_max_value, max_min_value, min_max_value


def unify_points(file_data:dict, iters:list, plot_targets:tuple) -> dict:
  new_iters = []
  result_file = {'iter':new_iters}
  for pt in plot_targets:
    file_idx = 0
    file_iters = file_data['iter']
    file_values = file_data[pt]
    result_values = []
    result_file[pt] = result_values
    for iter in iters:
      if file_idx < len(file_iters):
        result_values.append(file_values[file_idx])
        new_iters.append(iter)
        #since the iters passed in contain all possible iters we are either on this iter or ahead of it by some amount.
        if file_iters[file_idx] == iter:
          #looks like we were on it. Advance to the next one
          file_idx += 1
      else:
        break
  return result_file

def _get_iters(files_data:dict) -> (list, str, str):
  found_iters = set()
  longest = None
  longest_len = None
  shortest = None
  shortest_len = None
  #Find all the 'iters' in every file so we can build values for all files.
  for name, data in files_data.items():
    found_iters.update(data['iter'])
    if longest is None or longest_len < data['iter'][-1]:
      longest = name
      longest_len = data['iter'][-1]
    if shortest is None or shortest_len > data['iter'][-1]:
      shortest = name
      shortest_len = data['iter'][-1]
  iters = list(found_iters)
  iters.sort()
  return dict(iters=iters, longest=longest, shortest=shortest)
def _diff_to_target(files_data:dict, target:str, plot_targets:tuple) -> dict:
  target = files_data[target]
  result_files = dict()
  #Then we subtract the longest one's value from their values to normalize against it.
  for file_name, data in files_data.items():
    result_file = {'iter':data['iter'].copy()}
    result_files[file_name] = result_file
    for pt in plot_targets:
      target_values = []
      result_file[pt] = target_values
      for i in range(len(data['iter'])):
        if len(target[pt]) > i:
          target_values.append(data[pt][i] - target[pt][i])
        else:
          #Looks like the target is shorter than this run. We should trim it and not display the rest.
          result_file['iter'] = result_file['iter'][:i]
          break
  return result_files

def get_file_data(*files, plot_targets=('loss', 'accuracy')) -> (dict, dict):
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
      data['iter'] = []
      for row in csv_data:
        iter_val = int(row['iter'])
        if iter_val != 0:
          for data_name in plot_targets:
            value = float(row[data_name])
            data[data_name].append(value)
          data['iter'].append(iter_val)
      if len(data['iter']) > 0:
        file_data[name] = data
  return file_data, _get_iters(file_data)

def plot(out_file,
         file_data:(dict, dict),
         min_show=.1,
         log_plot=True,
         show=False,
         plot_targets=('loss',),
         scale=True,
         average_count:Optional[int] = 10,
         diff_to_target=False,
         use_process=True,
         target=None):
  out_file = os.path.expanduser(out_file)
  # We want to center the graph on the interesting areas so we need to track the overall min/max and the worst min value
  # across all datasets we are plotting. Then we will show from the absolute min to abve the worst min but below the abs max.

  file_data, file_meta = file_data
  #iters, longest, shortest = _get_iters(file_data)
  #now we have a spot for every iter. Let's get normalized value for every point by going through them all.
  #First we make sure that they all have values for every iter
  file_data = {name:unify_points(fd, file_meta['iters'], plot_targets) for name, fd in file_data.items()}
  min_iter = int(file_data[file_meta['shortest']]['iter'][-1]*(1.0 - min_show))
  #min_iter = iters[0]
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
       average_count)
