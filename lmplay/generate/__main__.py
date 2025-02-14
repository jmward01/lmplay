from lmplay import MODEL_RUNNERS
from lmplay.base.base_model import LMRunnerBase
import os


def main():
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


  results = mr.generate([prompt], max_len=args.max_len)[0]
  print(results)



if __name__ == "__main__":
  main()
