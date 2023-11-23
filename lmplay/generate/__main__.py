from lmplay.base.encoder.model import ModelRunner as BaseModelRunner
from lmplay import CurrentExpModelRunner as ModelRunner
import os


def main():
  from argparse import ArgumentParser
  args = ArgumentParser('Generates text!')
  args.add_argument('prompt', help="Prompt to start generation from")
  args.add_argument('--device', help="What device to use. default is CPU. 'cuda' and 'mps' are likely choices.", default='cpu')
  args.add_argument('--model', help="Location of the model file. Default is gpt_model.lmp.", default='gpt_model.lmp')
  args.add_argument('--exp', help="Use exp model runner. Changes regularly.", action="store_true")
  args.add_argument('--amp', help="Use Automatic Mixed Precision (AMP) training.", action="store_true")
  args.add_argument('--max-len', help="Set the max generation length. Default is model's max length.", type=int, default=None)
  args = args.parse_args()


  if args.exp:
    mr = ModelRunner()
  else:
    mr = BaseModelRunner()
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
  results = mr.generate([args.prompt], max_len=args.max_len)[0]
  print(results)



if __name__ == "__main__":
  main()
