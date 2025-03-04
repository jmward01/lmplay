from .plot import plot, find_stat_files, get_file_data
from argparse import ArgumentParser

def main():
  args = ArgumentParser("Graph some things!")
  args.add_argument('locations', help="Directories to look for training stats. Default is just out_gpt", nargs="*", default=["out_gpt"])
  args.add_argument('--plot-train', help="Generate plots for train stats instead of validate.", action="store_true")
  args.add_argument('--average-count', help="Smoothing factor. Default = 300", type=int, default=300)
  args.add_argument('--min-show', help="Min pct to show of the shortest run. Default is 0.99", type=float, default=0.99)
  args.add_argument('--out-type', help="type of plot to make either jpg, png. default is png", default='png')
  args.add_argument('--max-files', help="Max files to gather stats from. default is 20", default=20, type=int)
  args.add_argument('--target', help="What result to target. Only applies to the diff plots. default will use the longest available.", default=None)
  args.add_argument('--plot-raw', help="Plot the raw scatter info as well as the trend lines.", action="store_true")
  args = args.parse_args()
  if args.plot_train:
    stats_type = "train"
  else:
    stats_type = "validate"
  average_count = args.average_count
  max_files = args.max_files
  min_show = args.min_show

  file_directories = args.locations
  file_type = args.out_type

  found_files = []
  found_baseline_files = []
  for location in file_directories:
    files, baseline_files = find_stat_files(location)
    found_files.extend(files)
    found_baseline_files.extend(baseline_files)
  found_files = found_baseline_files + found_files


  validate_files = tuple(f for f in found_files if 'validate' in f)
  train_files = tuple(f for f in found_files if not 'validate' in f)

  if stats_type == "validate":
    found_files = validate_files
  elif stats_type == "train":
    found_files = train_files
  else:
    #Probably should have an option for train and validate
    raise ValueError(f"Unknown stat type {stats_type}")

  ignored_files = found_files[max_files:]
  found_files = found_files[:max_files]

  if len(found_files) == 0:
    print("No files found.")
    exit(1)
  print(f"Igrnoring: \n" + '\n'.join(ignored_files))
  print(f"Displaying: \n" + '\n'.join(found_files))

  file_data = get_file_data(*found_files)
  outfile_log_loss = f"./log_loss.{file_type}"
  outfile_log_acc = f"./log_accuracy.{file_type}"
  plot(outfile_log_loss, file_data, min_show=min_show, log_plot=True, scale=True, use_process=False, average_count=average_count, plot_raw=args.plot_raw)
  plot(outfile_log_acc, file_data, min_show=min_show, log_plot=True, scale=True, use_process=False, plot_targets=('accuracy',), average_count=average_count, plot_raw=args.plot_raw)

  #outfile_loss = f"./loss.{file_type}"
  #outfile_acc = f"./accuracy.{file_type}"
  #plot(outfile_loss, *found_files, min_show=.3, log_plot=False, scale=True, use_process=False)
  #plot(outfile_acc, *found_files, min_show=.9, log_plot=False, scale=False, use_process=False, plot_targets=('accuracy',))

  outfile_log_norm_loss = f"./log_diff_loss.{file_type}"
  outfile_log_norm_acc = f"./log_diff_accuracy.{file_type}"
  plot(outfile_log_norm_loss, file_data, min_show=min_show, log_plot=True, scale=True, use_process=False, diff_to_target=True, average_count=average_count, target=args.target, plot_raw=args.plot_raw)
  plot(outfile_log_norm_acc, file_data, min_show=min_show, log_plot=True, scale=True, use_process=False, plot_targets=('accuracy',), diff_to_target=True, average_count=average_count, target=args.target, plot_raw=args.plot_raw)



  #plot(outfile, *found_files, plot_targets=('accuracy',),min_show=.2, log_plot=True, scale=True, use_process=False, average_count=15)

if __name__ == "__main__":
  main()