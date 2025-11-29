"""
Main training entry point for lmplay language models.

This module provides the command-line interface for training language models with
various configurations and experimental setups. It handles:

- Multi-stage training plans with different datasets
- Model initialization and resumption from checkpoints
- Gradient accumulation for memory-efficient training
- Automatic Mixed Precision (AMP) training
- Validation and statistics tracking
- Model compilation optimizations
- Optimizer warmup for fine-tuning scenarios

The trainer supports various experimental model architectures through the runner
system and can work with multiple datasets including Wikipedia, OpenOrca, and
custom HuggingFace datasets.

Example usage:
    python -m lmplay.train --amp --device cuda --exp gpt2ish
    python -m lmplay.train --amp --device cuda --training-plan full
"""

import os.path, json
import logging

from lmplay.train.datasets.utils import batcher
from lmplay.stats.modelstats import ModelStats
import traceback
from tqdm import tqdm
from lmplay import MODEL_RUNNERS
from lmplay.base.base_runner import LMRunnerBase
from lmplay.base.exceptions import ModelCorrupted
from lmplay.train.datasets.plan import steps
from lmplay import config

logger = logging.getLogger('lmplay.train')

def render_pbar(exp: str, device, ms: ModelStats, ss: ModelStats, current_step: str) -> str:
  """
  Render progress bar description with training statistics.
  
  Args:
    exp: Experiment name
    device: Training device (cuda, cpu, mps)
    ms: Overall model statistics across all steps
    ss: Current step statistics
    current_step: Name of current training step
    
  Returns:
    Formatted string for progress bar description
  """
  if device is None:
    device = ""
  if ss.total_train_samples > 0:
    train_loss = f"{ss.train_loss():0.4f}"
    train_acc = f"{ss.train_accuracy():0.4f}"
  else:
    train_loss = "TBD"
    train_acc = "TBD"
  if ss.total_validate_samples > 0:
    validate_loss = f"{ss.validate_loss():0.4f}"
    validate_acc = f"{ss.validate_accuracy():0.4f}"
  else:
    validate_loss = "TBD"
    validate_acc = "TBD"
  b_step_tokens_trained = ss.total_train_tokens / 10e8
  b_tokens_trained = ms.total_train_tokens / 10e8
  return f"{exp}-{device}-{current_step}-train l:{train_loss}, a:{train_acc}/val l:{validate_loss}, a:{validate_acc}, st:{b_step_tokens_trained:0.2f}B, tt:{b_tokens_trained:0.2f}B"


def calc_next(interval: int, current: int) -> int:
  """
  Calculate the next checkpoint based on interval and current position.

  Args:
    interval: Checkpoint interval (e.g., save every 10000 samples)
    current: Current sample count

  Returns:
    Next checkpoint position
  """
  if current == 0:
    return interval
  return current + (interval - current % interval)


def log_training_configuration(args, mr, mini_batch_size, device):
  """
  Log comprehensive training configuration details.

  Args:
    args: Parsed command line arguments
    mr: Model runner instance
    batch_size: Effective batch size
    validation_batch_size: Validation batch size
    mini_batch_size: Mini batch size (GPU memory limited)
    device: Training device
  """
  print("=" * 70)
  print("TRAINING CONFIGURATION")
  print("=" * 70)

  # Model configuration
  print(f"Model: {mr._model.name}")
  print(f"Total Parameters: {mr._model.parameter_count():,} ({mr._model.parameter_count() / 1e9:.3f}B)")

  # Device configuration
  print(f"Device: {device}")
  print(f"AMP Enabled: {args.amp}")
  if args.compile_model:
    print(f"Model Compilation: Enabled (backend={args.compile_backend}, mode={args.compile_mode})")

  # Optimizer configuration
  if mr._optimizers:
    num_optimizers = len(mr._optimizers) if isinstance(mr._optimizers, list) else 1
    optimizer_list = mr._optimizers if isinstance(mr._optimizers, list) else [mr._optimizers]

    print(f"Optimizer: {num_optimizers} instance(s)")

    for opt_idx, optimizer in enumerate(optimizer_list):
      opt_lr = optimizer.param_groups[0].get('lr') if optimizer.param_groups else 'N/A'
      opt_weight_decay = optimizer.param_groups[0].get('weight_decay') if optimizer.param_groups else 'N/A'
      print(f"  Optimizer {opt_idx}:")
      print(f"    Learning Rate: {opt_lr}")
      print(f"    Weight Decay: {opt_weight_decay}")
      print(f"    Parameter Groups: {len(optimizer.param_groups)}")

      for group_idx, param_group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in param_group['params'])
        decay = param_group.get('weight_decay', 'N/A')
        print(f"      Group {group_idx}: {num_params:,} parameters, decay={decay}")

  # Batch configuration
  print(f"Effective Batch Size: {mr.batch_size}")
  print(f"Mini Batch Size: {mini_batch_size}")
  print(f"Gradient Accumulation Steps: {mr.batch_size // mini_batch_size}")
  print(f"Validation Batch Size: {mr.validation_batch_size}")

  # Training intervals
  print(f"Validation Interval: {mr.validation_interval} samples")
  print(f"Save Interval: {mr.save_interval} samples")

  # Gradient clipping
  if mr.grad_clip:
    print(f"Gradient Clipping: {mr.grad_clip}")

  # Model save location
  print(f"Model Save Location: {args.model}")

  print("=" * 70)


def main():
  """
  Main training function that handles argument parsing and training loop.
  
  This function:
  1. Parses command line arguments for training configuration
  2. Initializes the model runner with specified experiment
  3. Loads or creates training datasets according to the training plan
  4. Executes multi-stage training with validation and checkpointing
  5. Handles graceful shutdown on interruption
  
  The training loop supports:
  - Multi-stage training plans (e.g., pretraining -> fine-tuning)
  - Gradient accumulation for effective larger batch sizes
  - Regular validation with example output
  - Automatic model saving at specified intervals
  - Resume training from interruptions
  """
  from argparse import ArgumentParser
  args = ArgumentParser('Trains a GPT style model!')
  # while 'mps' works I have rarely seen it help and often seen it slow things down on mac.
  # This really surprises me since you would think Apple would be dumping massive resources into this so that their hardware would become the 'standard' for ML dev.
  # 'cuda' clearly rocks especially when used with AMP. That being said, this code is not built for multi-gpu since it is trying to be as simple as possible.
  args.add_argument('--device', help="What device to use. default is CPU. 'cuda' and 'mps' are likely choices.",
                    default=config._DefaultValue('cpu'))
  # Gradient accumulation is the only practical approach to training especially when you only have a 3070 to play with.
  # a mini-batch-size of 4 leaves a comfortable amount of extra RAM to try different things with a context length of 1024 and 12GB of RAM on the card.
  args.add_argument('--mini-batch-size', help="Mini batch size to use. Default is 4", default=config._DefaultValue(4), type=int)
  args.add_argument('--save-datasets',
                    help="Save the datasets to disk in the LMP_DATASETS directory or out_gpt/datasets if that env var isn't set then exit. This makes it easy to copy the data to another machine. You probably want to delete the ~/.cache/huggingface/datasets directory after this. Some training plans can take up .5TB or more of space in there.",
                    action="store_true", default=config._DefaultValue(False))
  #
  # pytorch's compile stuff is pretty cool, but still clearly buggy. These options are here for testing its impacts.
  # None of this is turned on for the published runs.
  args.add_argument('--compile-model', help="Ccompile the model before training.", action='store_true', default=config._DefaultValue(False))
  args.add_argument('--compile-mode',
                    help="Model for compiling. default is 'default'. Options are default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs.",
                    default=config._DefaultValue(None))
  args.add_argument('--compile-backend', help="Backend for compiling. default is 'inductor'.", default=config._DefaultValue("inductor"))
  #
  # AMP rocks. Use it. There is a minor hit in training but the speed and memory gains are completely worth it.
  # That being said, AMP and mac = not great.
  #
  args.add_argument('--amp', help="Use Automatic Mixed Precision (AMP) training.", action="store_true", default=config._DefaultValue(False))
  args.add_argument('--model', help="Model name to load/save to. Default is <exp>_<num_blocks>_<run_name>_model.lmp",
                    default=config._DefaultValue(None))
  args.add_argument('--initial-model',
                    help="Model file to look for if the 'model' isn't found. This model will only ever be read, not writen over. Default is gpt_initial_model.lmp.",
                    default=config._DefaultValue("gpt_initial_model.lmp"))
  args.add_argument('--exp',
                    help="Use exp model runner. Changes regularly. 'list' to show available models. default is gpt2ish",
                    default=config._DefaultValue("gpt2ish"))
  args.add_argument('--no-grad-scale',
                    help="only used with amp on cuda devices. Don't scale the grads. Only useful if using mixed devices (cpu and gpu)",
                    action="store_true", default=config._DefaultValue(False))
  args.add_argument('--check-grads', help="Prints any None gradients found while training.", action="store_true", default=config._DefaultValue(False))
  args.add_argument('--describe', help="Prints the model description and exits", action="store_true", default=config._DefaultValue(False))
  args.add_argument('--dump-config', help="Dump config file with construction_args and commented state_args_overrides, then exit.", default=config._DefaultValue(None), type=str)
  args.add_argument('--use-config', help="Load config file to merge with checkpoint state.", default=None, type=str)
  args.add_argument('--verbose', help="Enable verbose logging (DEBUG level). Default is INFO level.", action='store_true', default=config._DefaultValue(False))
  args = args.parse_args()

  # Load config BEFORE runner initialization/any of the args are used
  try:
    construction_args, state_args_overrides, run_args = config.load_config(args, args.use_config)
  except config.ConfigError as e:
    print(f"Error loading config: {e}")
    exit(1)

  # Configure logging based on verbose flag
  log_level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )

  # Suppress verbose logging from third-party libraries to avoid debug output spam
  logging.getLogger('matplotlib').setLevel(logging.WARNING)
  logging.getLogger('PIL').setLevel(logging.WARNING)


  if args.exp not in MODEL_RUNNERS:
    all_exps = ', '.join(MODEL_RUNNERS)
    if args.exp == 'list':
      print(f"Choose from {all_exps}")
      exit(0)
    else:
      print(f"{args.exp} not found. Choose from {all_exps}")
      exit(1)

  if args.describe == True:
    runner_info = MODEL_RUNNERS[args.exp]
    print(f"{runner_info['long_name']}:{runner_info['description']}")
    exit(0)

  # Set default model name if not provided
  if args.model is None:
    args.model = f"{args.exp}_model.lmp"

  initial_locations = [args.model, args.initial_model]
  save_location = args.model
  device = args.device


  #this will need to come from the mr after it reads the config.
  #batch size is a hyper parameter that should be considered one of the 'construction_args' so not passed in on the command line
  #minibatch size though is not a hyperparameter. If it is implemented correctly then it is just a performance parameter that allows smaller GPUs to train like bigger ones.
  #same with amp, compile_model, backend, etc. Those are 'command line' so would be available in the run_args/args

  construction_args['model']['version'] = args.exp
  mr:LMRunnerBase = MODEL_RUNNERS[args.exp]['runner'](mini_batch_size=args.mini_batch_size)
  mr.initialize(
    construction_args=construction_args,
    state_args_overrides=state_args_overrides,
    device=device,
    locations=initial_locations,
    for_train=True,
    compile_model=args.compile_model,
    compile_mode=args.compile_mode,
    compile_backend=args.compile_backend,
    amp=args.amp,
    no_grad_scale=args.no_grad_scale,
    check_grads=args.check_grads)

  # Handle --dump-config (save config and exit)
  if args.dump_config:
    construction_args_to_dump = mr.get_construction_args_for_config()
    state_args_to_dump = mr.get_state_args_for_config()
    config.save_config(args.dump_config, run_args, construction_args_to_dump, state_args_to_dump, comment_state_args=True)
    print(f"Config file saved to: {args.dump_config}")
    exit(0)

  # Log training configuration if verbose is enabled
  log_training_configuration(args, mr, args.mini_batch_size, device)

  early_exit = False
  for step_name, epochs, train, validation in steps(mr.training_plan, current_step=mr.current_step):
    did_training = False
    mr.set_current_step(step_name)
    if save_location.endswith('.lmp'):
      step_save_location = f"{save_location[:-4]}.{step_name}.lmp"
    else:
      step_save_location = f"{save_location}.{step_name}"
    validate_interval = mr.validation_interval
    next_validate = calc_next(validate_interval, mr.get_step_stats().total_train_samples)

    save_interval = mr.save_interval
    next_save = calc_next(save_interval, mr.get_step_stats().total_train_samples)
    with tqdm(total=int(len(train)*epochs), initial=mr.get_step_stats().total_train_samples) as pbar:

      train_batcher = batcher(train,
                              batch_size=mr.batch_size,
                              epochs=epochs,
                              fast_forward=mr.get_step_stats().total_train_samples,
                              max_length=mr.max_len)

      validation_batcher = batcher(validation,
                                   batch_size=mr.validation_batch_size,
                                   fast_forward=mr.get_step_stats().total_validate_samples,
                                   max_length=mr.max_len)
      total_parameters = mr._model.parameter_count()
      print(f"\nTraining {mr._model.name} with {total_parameters}({total_parameters / 1e9:0.3f}b) parameters.\n")
      try:

        for batch, new_train_samples_read in train_batcher:
          did_training = True
          # Hack because hugging face doesn't have a way to restart where you left off.
          # Trying to preserve order to make testing repeatable but still allow interruptions
          # if train_count > mr.model_stats.total_train_samples:
          results, _, total_tokens = mr.train(batch, new_train_samples_read)

          if mr.get_step_stats().total_train_samples >= next_validate:
            validation_batch, new_validation_samples_read = next(validation_batcher)
            results, _, validate_tokens = mr.validate(validation_batch, new_validation_samples_read)
            truth_example: str = validation_batch[-1]['truth']
            prediction_example: str = results[-1]
            prompt = validation_batch[-1]['prompt'].replace('\n', ' ')
            truth_example = truth_example.replace('\n', ' ')
            prediction_example = prediction_example.replace('\n', ' ')
            # print(f"\nSystem:{validation_batch[0]['system']}\nUser:{validation_batch[0]['user']}\nTruth/Prediction:\n{truth_example[:200]}\n{prediction_example[:200]}")
            print(f"\nPrompt:\n{prompt}\nTruth/Prediction:\n{truth_example}\n{prediction_example}")
            next_validate = calc_next(validate_interval, mr.get_step_stats().total_train_samples)

          if mr.get_step_stats().total_train_samples >= next_save:
            pbar.set_description("Saving weights...")
            mr.save(save_location)
            next_save = calc_next(save_interval, mr.get_step_stats().total_train_samples)
          pbar.set_description(render_pbar(args.exp, device, mr.model_stats, mr.get_step_stats(), mr.current_step))
          pbar.update(new_train_samples_read)
      except ModelCorrupted as e:
        print(f"\n{'='*70}")
        print("MODEL CORRUPTION DETECTED - TRAINING HALTED")
        print(f"{'='*70}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        print(f"\nCorruption details:\n{e.format_details()}")
        print(f"{'='*70}")
        # Save corrupted state for analysis
        if save_location.endswith('.lmp'):
          corrupt_save_location = f"{save_location[:-4]}.corrupted.lmp"
        else:
          corrupt_save_location = f"{save_location}.corrupted"
        print(f"Saving corrupted model state to: {corrupt_save_location}")
        mr.save(corrupt_save_location)
        early_exit = True
      except KeyboardInterrupt:
        print(f"User canceled training.")
        early_exit = True
      except:
        print(f"Unknown error:\n{traceback.format_exc()}")
        early_exit = True
      if early_exit:
        break
      if did_training:
        #if they started on a finished step we likely loaded the data from that step but didn't train on it so don't overwrite those weights.
        pbar.set_description("Step ended. Saving final step weights...")
        mr.save(step_save_location)

if __name__ == "__main__":
  main()
