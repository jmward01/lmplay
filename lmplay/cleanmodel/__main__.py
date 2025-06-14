"""
Command-line interface for cleaning and preparing trained models.

This module provides the main entry point for model cleaning, accessible via
the 'lmp_cleanmodel' command. It processes trained models to remove 
training-specific parameters and prepare them for efficient deployment.

Usage:
  lmp_cleanmodel model.lmp --model-out clean_model.lmp
  lmp_cleanmodel model.lmp --exp

The cleaning process:
1. Loads the trained model using the appropriate runner
2. Removes training-specific parameters (sacrificial weights, etc.)
3. Converts complex architectures to inference-optimized forms
4. Saves a cleaned model suitable for production deployment

This is particularly useful for models trained with experimental features
like Unified Embeddings, sacrificial networks, or other training aids
that should be removed for deployment.
"""

from lmplay.base.encoder.model import ModelRunner as GPT2ModelRunner
from lmplay import CurrentExpModelRunner as ModelRunner
import os


def main():
  """Main entry point for model cleaning CLI.
  
  Parses command-line arguments, loads the specified model, and saves a cleaned
  version optimized for deployment. The cleaning process removes training-specific
  parameters and converts complex architectures to efficient inference forms.
  
  Command-line Arguments:
    model (str): Path to the trained model file (.lmp) to clean.
    --model-out (str): Output path for the cleaned model. Defaults to 
      'gpt_model_clean.lmp'.
    --exp: Use experimental model runner instead of default GPT2 runner.
      Should match the runner used during training.
  
  The function handles model loading, cleaning, and saving with proper error
  handling for missing files. The cleaned model will be suitable for production
  deployment with reduced memory usage and faster loading.
  """
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