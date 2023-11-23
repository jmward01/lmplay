from lmplay.base.encoder.model import ModelRunner as GPT2ModelRunner
from lmplay import CurrentExpModelRunner as ModelRunner
import os


def main():
  from argparse import ArgumentParser
  args = ArgumentParser('Removes all but the weights from a model.')
  args.add_argument('model', help="Model name to load from. Default is gpt_model.lmp", default="gpt_model.lmp")
  args.add_argument('--model-out', help="Model name to save to. Default is gpt_model_clean.lmp", default="gpt_model_clean.lmp")
  args.add_argument('--exp', help="Use exp model runner. Changes regularly.", action="store_true")
  args = args.parse_args()

  location = os.path.expanduser(args.model)
  if not os.path.exists(location):
    print(f"Unable to find {location}")
    exit(1)

  if args.exp:
    mr = ModelRunner()
  else:
    mr = GPT2ModelRunner()
  mr.initialize('cpu', locations=[args.model], for_train=False)
  mr.save(args.model_out, prod_save=True)


if __name__ == "__main__":
  main()