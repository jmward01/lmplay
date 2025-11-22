"""
Command-line interface for text generation with language models.

This module provides the main entry point for text generation, accessible via
the 'lmp_generator' command. It loads trained models and generates text based
on provided prompts.

Usage:
    lmp_generator "Your prompt here" --model model.lmp --device cuda
    lmp_generator prompt.txt --exp gpt2ish --max-len 500

The generator supports loading prompts from files and provides various
configuration options for controlling generation behavior.
"""

from lmplay import MODEL_RUNNERS
from lmplay.base.base_model import LMRunnerBase
import os


def main():
  """Main entry point for text generation CLI.

  Parses command-line arguments, loads the specified model, and generates
  text based on the provided prompt. Supports various configuration options
  including device selection, model type, and generation parameters.

  Command-line Arguments:
    prompt (str): Text prompt to start generation. Can be either direct text
      or a filename containing the prompt text.
    --device (str): Device to use for generation ('cpu', 'cuda', 'mps').
      Defaults to 'cpu'.
    --model (str): Path to the model file (.lmp). Defaults to 'gpt_model.lmp'.
    --exp (str): Experiment type to use. Use 'list' to see available options.
      Defaults to 'gpt2ish'.
    --amp: Enable Automatic Mixed Precision for generation.
    --max-len (int): Maximum generation length. Defaults to model's max_len.
    --temperature (float): Sampling temperature (default 1.0).
      < 1.0 = more deterministic, > 1.0 = more random.
    --top-k (int): Keep only top k tokens (default: disabled).
      Set to > 0 to enable.
    --top-p (float): Nucleus sampling threshold (default 1.0, disabled).
      Typical values: 0.9, 0.95
    --repetition-penalty (float): Penalty for repeated tokens (default 1.0).
      > 1.0 discourages repetition.
    --do-sample: Use sampling instead of greedy decoding.

  The function handles model loading, device configuration, and text generation
  with proper error handling for missing files and invalid experiments.
  """
  from argparse import ArgumentParser
  args = ArgumentParser('Generates text!')
  args.add_argument('prompt', help="Prompt text to start generation on. If this is a file name it will be loaded.")
  args.add_argument('--device', help="What device to use. default is CPU. 'cuda' and 'mps' are likely choices.", default='cpu')
  args.add_argument('--model', help="Location of the model file. Default is gpt_model.lmp.", default='gpt_model.lmp')
  args.add_argument('--exp',
                    help="Use exp model runner. Changes regularly. 'list' to show available models. default is gpt2ish",
                    default="gpt2ish")
  args.add_argument('--amp', help="Use Automatic Mixed Precision (AMP) training.", action="store_true")
  args.add_argument('--max-len', help="Set the max generation length. Default is model's max length.", type=int, default=None)

  # Generation control arguments
  args.add_argument('--temperature', help="Sampling temperature (default 1.0). < 1.0 = more deterministic, > 1.0 = more random.",
                    type=float, default=1.0)
  args.add_argument('--top-k', help="Keep only top k tokens (default: disabled). Set to > 0 to enable.", type=int, default=None)
  args.add_argument('--top-p', help="Nucleus sampling threshold (default 1.0, disabled). Typical values: 0.9, 0.95.",
                    type=float, default=1.0)
  args.add_argument('--repetition-penalty', help="Penalty for repeated tokens (default 1.0, no penalty). > 1.0 discourages repetition.",
                    type=float, default=1.0)
  args.add_argument('--do-sample', help="Use sampling instead of greedy decoding.", action="store_true")

  args = args.parse_args()


  if args.exp not in MODEL_RUNNERS:
    all_exps = ', '.join(MODEL_RUNNERS)
    if args.exp == 'list':
      print(f"Choose from {all_exps}")
      exit(0)
    else:
      print(f"{args.exp} not found. Choose from {all_exps}")
      exit(1)
  mr:LMRunnerBase = MODEL_RUNNERS[args.exp]['runner']()
  location = os.path.expanduser(args.model)
  if not os.path.exists(location):
    print(f"{location} not found.")
    exit(1)
  mr.initialize(args.device,
                locations=args.model,
                amp=args.amp,
                for_train=False)
  if args.max_len is None:
    args.max_len = mr._model.max_len
  total_parameters = mr._model.parameter_count()
  print(f"\nGenerating {mr._model.name} with {total_parameters}({total_parameters/1e9:0.3f}b) parameters.\n")
  prompt = args.prompt
  if not ' ' in args.prompt:
    if os.path.exists(os.path.expanduser(args.prompt)):
      with open(os.path.expanduser(args.prompt)) as infile:
        prompt = infile.read()


  results = mr.generate(
      [prompt],
      max_len=args.max_len,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      repetition_penalty=args.repetition_penalty,
      do_sample=args.do_sample)[0]
  print(results)



if __name__ == "__main__":
  main()
