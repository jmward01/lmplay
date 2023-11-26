from datasets import concatenate_datasets, interleave_datasets
from lmplay.train.lmpdatasets import get_wikipedia_en, get_wikipedia_simple
from lmplay.train.utils import batcher
from lmplay.stats.modelstats import ModelStats
import traceback
from typing import Optional
from tqdm import tqdm
from lmplay.base.encoder.model import ModelRunner as BaseModelRunner
from lmplay import CurrentExpModelRunner as ModelRunner


def render_pbar(ms: ModelStats) -> str:
  if ms.total_train_samples > 0:
    train_loss = f"{ms.train_loss():0.4f}"
    train_acc = f"{ms.train_accuracy():0.4f}"
  else:
    train_loss = "TBD"
    train_acc = "TBD"
  if ms.total_validate_samples > 0:
    validate_loss = f"{ms.validate_loss():0.4f}"
    validate_acc = f"{ms.validate_accuracy():0.4f}"
  else:
    validate_loss = "TBD"
    validate_acc = "TBD"
  return f"train l:{train_loss}, a:{train_acc}/val l:{validate_loss}, a:{validate_acc}"


def calc_next(interval: int, current: int):
  if current == 0:
    return interval
  return current + (interval - current % interval)


def main():
  from argparse import ArgumentParser
  args = ArgumentParser('Trains a GPT style model!')
  #while 'mps' wroks I have rarely seen it help and often seen it slow things down on mac.
  # This really surprises me since you would think Apple would be dumping massive resources into this so that their hardware would become the 'standard' for ML dev.
  # 'cuda' clearly rocks especially when used with AMP. That being said, this code is not built for multi-gpu since it is trying to be as simple as possible.
  args.add_argument('--device', help="What device to use. default is CPU. 'cuda' and 'mps' are likely choices.", default='cpu')
  #Gradient accumulation is the only practical approach to training especially when you only have a 3070 to play with.
  #a mini-batch-size of 4 leaves a comfortable amount of extra RAM to try different things with a context length of 1024 and 12GB of RAM on the card.
  args.add_argument('--mini-batch-size', help="Mini batch size to use. Default is 4", default=4 , type=int)
  #50 was mostly arbitrarily picked here. 'larger is better' is the mantra but 50does an ok job of training quickly but still training deeply.
  args.add_argument('--batch-size', help="Batch size to use. Default is 50", default=50, type=int)
  #Make validation-batch-size larger to get more smoothed out validation. The larger this is though the more compute taken from training.
  args.add_argument('--validation-batch-size', help="Batch size to use for validation. Default is 4", default=4 , type=int)
  #Similar to validation-batch-size, validation-interval allows you to get more fine-grained validation
  args.add_argument('--validation-interval', help="how many train samples before a validation batch. Default is 100.", default=100, type=int)
  args.add_argument('--save-interval', help="How many train samples until the model is saved. Default is 2000", default=2000, type=int)
  args.add_argument('--num-blocks', help="Number of layers to initialize a new model with. Default is 6.", default=6, type=int)
  args.add_argument('--epochs', help="Number of times to go over the total train set. Epochs are evil but easy to implement. Default = 1.0", default=1.0, type=float)
  args.add_argument('--default-freeze', help="Freeze all weights by default unless '.freeze=False' on them or a module they are a part of.", action='store_true')
  args.add_argument('--ignore-optimizer', help="don't load optimizer weights if found.", action='store_true')
  args.add_argument('--lr', help="Learning rate. Default is left up to the model. 0.0006 is normally ok.", default=None, type=float)
  args.add_argument('--run-name', help="Run name to add to model stats. Useful for distinguishing runs with different datasets. default = 'simple_full_wiki'", default='simple_full_wiki')


  #What is 'optimizer warmup' you wonder?
  # Well, depending on the optimizer used and what the model was trained on you can experience severe 'model shock' if a cold optimizer is used against well trained weights.
  # If you have ever fine-tuned a model only to see its performance get worse this is a possible cause.
  #
  # Model shock can be seen clearly in loss data.
  # Its signature is an initial batch that has normal loss followed by a sudden increase in loss and then a slow steady drop back to normal.
  # That sudden peak likely deeply damaged the well trained weights
  #
  #You can drastically cut down on model shock by cutting the lr way down initially and then turning it up slowly.
  #that is what the 'optimizer-warmup-fraction' stuff is all about. It is here for tests where the optimizer state is lost (because of weight size changes or other reasons).
  #It really isn't needed for most situations though.
  #
  args.add_argument('--optimizer-warmup-fraction', help="Sets the starting lr in fraction of final lr for warming up the lr if model weights are found but optimizer weights aren't there. default is 0.1 of the final lr.", default=None, type=float)
  args.add_argument('--optimizer-warmup-steps', help='Sets the number of steps for optimizer warmup if no optimizer weights are present but there are model weights. default is 40 batches.', default=None, type=int)
  args.add_argument('--disable-optimizer-warmup', help='Turns off optimizer warmup.', action='store_true')
  #
  #pytorch's compile stuff is pretty cool, but still clearly buggy. These options are here for testing its impacts.
  # None of this is turned on for the published runs.
  args.add_argument('--compile-model', help="Ccompile the model before training.", action='store_true')
  args.add_argument('--compile-mode', help="Model for compiling. default is 'default'. Options are default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs.", default=None)
  args.add_argument('--compile-backend', help="Backend for compiling. default is 'inductor'.", default="inductor")
  #
  #AMP rocks. Use it. There is a minor hit in training but the speed and memory gains are completely worth it.
  #That being said, AMP and mac = not great.
  #
  args.add_argument('--amp', help="Use Automatic Mixed Precision (AMP) training.", action="store_true")
  args.add_argument('--model', help="Model name to load/save to. Default is gpt_model.lmp", default="gpt_model.lmp")
  args.add_argument('--initial-model', help="Model file to look for if the 'model' isn't found. This model will only ever be read, not writen over. Default is gpt_initial_model.lmp", default="gpt_initial_model.lmp")
  args.add_argument('--exp', help="Use exp model runner. Changes regularly.", action="store_true")
  args.add_argument('--no-grad-scale', help="only used with amp on cuda devices. Don't scale the grads. Only useful if using mixed devices (cpu and gpu)", action="store_true")
  args = args.parse_args()
  if args.amp and args.compile_model:
    #I haven't really gotten this to work and haven't really spent a lot of time trying.
    # Leaving it in for possible future runs.
    print("WARNING: AMP is being used with a compiled model. This generally has issues and could be very slow. You have been warned!")


  initial_locations = [args.model, args.initial_model]
  save_location = args.model
  device = args.device
  #datasets = get_wikipedia_simple(seed=seed)
  seed = 0
  full_datasets = get_wikipedia_en(seed=seed)
  simple_datasets = get_wikipedia_simple(seed=seed)
  train = concatenate_datasets([simple_datasets['train'], full_datasets['train']])
  validation = interleave_datasets([simple_datasets['validation'], full_datasets['validation']])
  batch_size = args.batch_size
  validation_batch_size = args.validation_batch_size
  mini_batch_size = min(batch_size, args.mini_batch_size)




  if args.exp:
    mr = ModelRunner(max_batch_size=mini_batch_size)
  else:
    mr = BaseModelRunner(max_batch_size=mini_batch_size)
  mr.initialize(device,
                locations=initial_locations,
                run_name=args.run_name,
                num_blocks=args.num_blocks,
                default_freeze=args.default_freeze,
                load_optimizer=not args.ignore_optimizer,
                optimizer_warmup_fraction=args.optimizer_warmup_fraction,
                optimizer_warmup_steps=args.optimizer_warmup_steps,
                disable_optimizer_warmup=args.disable_optimizer_warmup,
                lr=args.lr,
                compile_model=args.compile_model,
                compile_mode=args.compile_mode,
                compile_backed=args.compile_backend,
                amp=args.amp,
                no_grad_scale=args.no_grad_scale)
  epochs = args.epochs

  validate_interval = args.validation_interval
  next_validate = calc_next(validate_interval, mr.model_stats.total_train_samples)

  save_interval = args.save_interval
  next_save = calc_next(save_interval, mr.model_stats.total_train_samples)

  pbar = tqdm(total=len(train), initial=mr.model_stats.total_train_samples)

  def to_row(row: dict) -> (dict, Optional[int]):
    #user = f'Write an article with the following title: {row["title"]}'
    #system = 'You write wikipedia articles.'
    #truth = f'Title: {row["title"]}. Article: {row["text"]}'
    prompt = ""
    #return {'system': system, 'user': user, 'truth': truth}
    if 'title' in row:
      prompt = f'Title: {row["title"]}\n'
    else:
      prompt = ''
    truth:str = row['text']
    ts = truth.split()
    ps = prompt.split()
    estimated_tokens = (len(ps) + len(ts))*1.5
    continuation = None
    if estimated_tokens > mr.max_len:
      #Find the number of words into the truth (approximately) to start looking for a split.
      start_trim = int(mr.max_len/1.4 - len(ps)*1.5)
      #Now convert that ti an actual place in the string
      start_trim = len(' '.join(ts[:start_trim]))
      nearest_period = truth.find('. ', start_trim)
      if nearest_period > 0 and nearest_period  + 2 < len(row['text']):
        continuation = nearest_period + 2
    return {'prompt':prompt, 'truth': truth}, continuation

  def continue_row(row:dict, continuation:int) -> (dict, Optional[int]):
    next_nearest_period = row['text'].find('. ', continuation)
    if 'title' in row:
      prompt = f'Title: {row["title"]}\n...'
    else:
      prompt = '...'
    if next_nearest_period > 0 and next_nearest_period + 2 < len(row['text']):
      prompt = prompt + row['text'][continuation:next_nearest_period + 2]
      continuation = next_nearest_period + 2

    truth = row['text'][continuation:]
    ts = truth.split()
    ps = prompt.split()
    estimated_tokens = (len(ps) + len(ts))*1.5
    new_continuation = None
    if estimated_tokens > mr.max_len:
      #Find the number of words into the truth (approximately) to start looking for a split.
      start_trim = int(mr.max_len/1.4 - len(ps)*1.5)
      #Now convert that ti an actual place in the string
      start_trim = len(' '.join(ts[:start_trim]))
      nearest_period = truth.find('. ', start_trim)
      if nearest_period > 0 and nearest_period + continuation + 2 < len(row['text']):
        new_continuation = nearest_period + continuation + 2
    return {'prompt':prompt, 'truth': truth}, new_continuation


  train_batcher = batcher(train,
                          batch_size=batch_size,
                          row_reader=to_row,
                          row_continuation=continue_row,
                          epochs=epochs,
                          fast_forward = mr.model_stats.total_train_samples)

  validation_batcher = batcher(validation,
                               batch_size=validation_batch_size,
                               row_reader=to_row,
                               row_continuation=continue_row,
                               fast_forward=mr.model_stats.total_validate_samples)
  total_parameters = mr._model.parameter_count()
  print(f"\nTraining {mr._model.name} with {total_parameters}({total_parameters/10e9:0.3f}b) parameters.\n")
  try:
    for batch, new_train_samples_read in train_batcher:
      # Hack because hugging face doesn't have a way to restart where you left off.
      # Trying to preserve order to make testing repeatable but still allow interruptions
      # if train_count > mr.model_stats.total_train_samples:
      results, _ = mr.train(batch, new_train_samples_read)

      if mr.model_stats.total_train_samples >= next_validate:
        validation_batch, new_validation_samples_read = next(validation_batcher)
        results, _ = mr.validate(validation_batch, new_validation_samples_read)
        truth_example: str = validation_batch[-1]['truth']
        prediction_example: str = results[-1]
        prompt = validation_batch[-1]['prompt'].replace('\n', ' ')
        truth_example = truth_example.replace('\n', ' ')
        prediction_example = prediction_example.replace('\n', ' ')
        #print(f"\nSystem:{validation_batch[0]['system']}\nUser:{validation_batch[0]['user']}\nTruth/Prediction:\n{truth_example[:200]}\n{prediction_example[:200]}")
        print(f"\nPrompt:\n{prompt}\nTruth/Prediction:\n{truth_example}\n{prediction_example}")
        next_validate = calc_next(validate_interval, mr.model_stats.total_train_samples)

      if mr.model_stats.total_train_samples >= next_save:
        pbar.set_description("Saving weights...")
        mr.save(save_location)
        next_save = calc_next(save_interval, mr.model_stats.total_train_samples)
      pbar.set_description(render_pbar(mr.model_stats))
      pbar.update(new_train_samples_read)
  except KeyboardInterrupt:
    print(f"User canceled training.")
  except:
    print(f"Unknown error:\n{traceback.format_exc(limit=10)}")



if __name__ == "__main__":
  main()
