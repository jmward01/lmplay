import os.path
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Any, List
import torch
from torch import nn
from lmplay.stats import modelstats, utils
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.utils.rnn import pad_sequence
from shutil import copyfile
import torch.nn.functional as F

DEFAULT_LR = 6e-4
# DEFAULT_LR = 3e-5
DEFAULT_WEIGHT_DECAY = 0.00


class OptimizerWarmupLRScheduler(LRScheduler):
  def __init__(self, optimizer: Optimizer, steps: Optional[int] = 100, initial_fraction: Optional[float] = 0.2):
    steps = steps if steps else 40
    initial_fraction = initial_fraction if initial_fraction else 0.1

    self.increasing = initial_fraction < 1.0
    self.max_lrs = [group['lr'] for group in optimizer.param_groups]
    self.current_lrs = [lr * initial_fraction for lr in self.max_lrs]
    self.step_size = [(m - c) / steps for m, c in zip(self.max_lrs, self.current_lrs)]
    # for the initial call on the first batch.
    self.current_lrs = [c - s for c, s in zip(self.current_lrs, self.step_size)]
    super().__init__(optimizer)

  def get_lr(self):
    if self.increasing:
      next_lr = [min(m, c + s) for m, c, s in zip(self.max_lrs, self.current_lrs, self.step_size)]
    else:
      next_lr = [max(m, c + s) for m, c, s in zip(self.max_lrs, self.current_lrs, self.step_size)]
    self.current_lrs = next_lr
    return self.current_lrs


class LMBase(nn.Module):
  def __init__(self, name: str, *init_args, **init_kwargs):
    super().__init__()
    self.name = name.replace('.', '_')
    self.init_args = init_args
    self.init_kwargs = init_kwargs
    self.max_len = init_kwargs['max_len']

  def initialize(self, device):
    pass
    # self.unified_tok_embed.initialize(self.tok_embed, device)

  def _kv_cache(self, cache: Optional[list], idx):
    if cache is None:
      return None
    if len(cache) <= idx:
      cache.append([])
    return cache[idx]

  def _tokenize_str(self, sample: dict, device, trim=True) -> (torch.Tensor, int):
    prompt = sample['prompt']
    tokens = [self.tokenizer.eot_token] + self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    # adjusting because the the model is - 1 on its prediction.
    # for example if the model sees this (& = bos/eos):
    # & P R E D I C
    # It will predict like this:
    # P R E D I C T
    # So it we said the prompt ended on 1 then prediction starts on 0
    prediction_starts = len(tokens) - 1
    if 'truth' in sample:
      truth: str = sample['truth']
      if not truth.endswith("<|endoftext|>"):
        tokens.extend(self.tokenizer.encode(sample['truth'] + "<|endoftext|>", allowed_special={"<|endoftext|>"}))
      else:
        tokens.extend(self.tokenizer.encode(sample['truth'], allowed_special={"<|endoftext|>"}))

    # We can go one more because one is being trimmed off
    if trim and len(tokens) > self.max_len + 1:
      # too long. Just cut it off
      tokens = tokens[:self.max_len + 1]
      # tokens[-1] = self.tokenizer.eot_token
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    return tokens, prediction_starts

  def _tokenize_batch(self, batch: Sequence[dict]) -> (torch.Tensor, Sequence[int]):
    device = self.fc.weight.device
    predictions_starts = []
    predictions_ends = []
    x = []
    for t in batch:
      t, ps = self._tokenize_str(t, device)
      x.append(t)
      predictions_starts.append(ps)
      predictions_ends.append(t.size(-1) - 1)

    x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.eot_token)
    return x, predictions_starts, predictions_ends

  def to(self, *args, **kwargs):
    # Modified from pytorch 2.1 source
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    def convert(t):
      if convert_to_format is not None and t.dim() in (4, 5):
        if hasattr(t, 'force_device'):
          return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None,
                      non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking, memory_format=convert_to_format)
      if hasattr(t, 'force_device'):
        return t.to(t.force_device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
      return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)

  @abstractmethod
  def forward(self, x: torch.Tensor, cache: Optional[List] = None):
    pass

  def train_prompts(self, prompts: Sequence[dict]) -> (Sequence[str], torch.Tensor):
    # We want to pad them together so that the truths will line up with the prompts.
    x, predictions_starts, predictions_ends = self._tokenize_batch(prompts)
    # Truth doesn't have the first EOT char. It needs to start on prediction start
    truths = x[:, 1:]
    # x doesn't need the last EOT since it will be predicting that
    x = x[:, :-1]

    x = self(x)
    results = []
    # num classes is always second. For, reasons?
    target_loss = F.cross_entropy(x.permute(0, 2, 1), truths, reduction="none")
    total_target_loss = 0.0
    for tl, prediction_start, prediction_end in zip(target_loss, predictions_starts, predictions_ends):
      tl = tl[prediction_start:prediction_end]
      tl = tl.sum()
      token_count = max(prediction_end - prediction_start, 1)
      # norm by number of tokens in the truth
      total_target_loss = tl / token_count + total_target_loss
    target_loss = total_target_loss
    # Get the predicted value so we can cut just that out as the result.
    predicted_tokens = torch.argmax(x, dim=-1)
    # for result, prediction_start, prediction_end, truth in zip(predicted_tokens, predictions_starts, predictions_ends, truths):
    for result, prediction_start, prediction_end in zip(predicted_tokens, predictions_starts, predictions_ends):
      # we only care about the prediction.
      # the last value is end of sentence
      results.append(self.tokenizer.decode(result[prediction_start:prediction_end].tolist()))
    # target loss is normalized by example but not by batch. That will be done by the caller.
    return results, target_loss

  def generate_prompts(self, prompts: Sequence[dict], max_len: Optional[int] == None) -> Sequence[str]:
    results = []
    if not max_len:
      max_len = self.max_len
    with torch.no_grad():
      for prompt in prompts:
        result = []
        # We only support batch size of 1. This code is just for testing and not meant to be fast/good/etc.
        # Basically, we want training to be -really- easy to play with so we have the minor concession of a kv cache mechanism.
        x, _, _ = self._tokenize_batch([prompt])
        cache = []
        stop = False
        while not stop:
          x, cache = self(x, cache=cache)
          # Should only be one token
          x = torch.argmax(x, dim=-1)
          result.append(x.squeeze().tolist())
          if result[-1] == self.tokenizer.eot_token or len(result) == max_len:
            stop = True
        results.append(self.tokenizer.decode(result))

    return results

  def parameter_count(self) -> int:
    pc = 0
    for p in self.parameters():
      p_count = 1
      for s in p.shape:
        p_count *= s
      pc += p_count
    return pc


def detect_freeze(module: nn.Module):
  for m in module.modules():
    if hasattr(m, 'freeze') and m.freeze is not None:
      freeze = m.freeze
      m.requires_grad_(not freeze)
  for p in module.parameters():
    if hasattr(p, 'freeze') and p.freeze is not None:
      freeze = p.freeze
      p.requires_grad_(not freeze)


class NopWith:
  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass


class LMRunnerBase(ABC):
  def __init__(self, max_batch_size: int = 1, stats_dir="./out_gpt"):
    self.max_batch_size = max_batch_size
    self._model: Optional[LMBase] = None
    self._raw_model: Optional[LMBase] = None
    self._optimizers: Optional[List[Optimizer]] = None
    self.model_stats: Optional[modelstats.ModelStats] = None
    self.step_stats = dict()
    self.current_step = None
    self._model_args = None
    self._optimizer_args = None
    self._stats_dir = stats_dir
    self._lr_scheduler: Optional[LRScheduler] = None
    self.run_name = ""
    self.device: Optional[str] = None
    self.device_type: Optional[str] = None
    self.max_len: Optional[int] = None

  def set_current_step(self, step_name: str):
    if not self.current_step is None and step_name != self.current_step:
      self.get_step_stats().write_train()
      self.get_step_stats().write_validate()
    self.current_step = step_name

  def is_trainable(self) -> bool:
    return self._optimizers is not None

  def is_initialzed(self) -> bool:
    return self._model is not None

  def get_step_stats(self) -> modelstats.ModelStats:
    if self.current_step not in self.step_stats:
      self.step_stats[self.current_step] = modelstats.ModelStats(
        model_name=f"{self._model.name}{self.run_name}_step_{self.current_step}",
        basedir=self._stats_dir)
    return self.step_stats[self.current_step]

  def initialize(self,
                 device,
                 locations: Optional[Union[Sequence[str], str]] = None,
                 for_train=True,
                 load_optimizer=True,
                 strict=False,
                 run_name="",
                 default_freeze=False,
                 optimizer_warmup_fraction: Optional[float] = None,
                 optimizer_warmup_steps: Optional[int] = None,
                 disable_optimizer_warmup=False,
                 compile_model=False,
                 compile_mode=None,
                 compile_backend='inductor',
                 amp=False,
                 no_grad_scale=False,
                 reset_history=False,
                 first_step=None,
                 grad_clip=None,
                 check_grads=False,
                 **parameters):
    self.check_grads = check_grads
    self.grad_clip = grad_clip
    self.for_train = for_train
    self.device = device
    if 'cuda' in self.device:
      self.device_type = "cuda"
    elif 'mps' in self.device:
      self.device_type = "mps"
    elif 'cpu' in self.device:
      self.device_type = "cpu"
    else:
      self.device_type = device
    # if torch.cuda.is_available():
    #  torch.set_float32_matmul_precision('high')
    torch.set_float32_matmul_precision('high')
    for p in ('lr', 'optimizer_warmup_start', 'optimizer_warmup_steps'):
      if p in parameters and parameters[p] is None:
        del parameters[p]

    self.step_stats = dict()
    self.current_step = first_step

    if len(run_name) > 0:
      self.run_name = f"_{run_name}"
    else:
      self.run_name = ""
    if locations is None:
      locations = []
    if isinstance(locations, str):
      locations = [locations]
    if locations is not None:
      locations = [os.path.expanduser(location) for location in locations]
      locations = [location for location in locations if os.path.exists(location)]

    if len(locations) > 0:
      location = locations[0]
      weight_data = torch.load(location, map_location=device)
      self._model, self._model_args, missing, unexpected = self._construct_model(device,
                                                                                 model_weights=weight_data.get('model',
                                                                                                               None),
                                                                                 model_args=weight_data.get(
                                                                                   'model_args',
                                                                                   None),
                                                                                 strict=strict,
                                                                                 **parameters)
      if reset_history:
        self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}",
                                                 basedir=self._stats_dir)

      else:
        self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}",
                                                 **weight_data.get('stats', {}),
                                                 basedir=self._stats_dir)
        self.current_step = weight_data.get('current_step', first_step)
        for step_name, data in weight_data.get('step_stats', dict()).items():
          self.step_stats[step_name] = modelstats.ModelStats(
            model_name=f"{self._model.name}{self.run_name}_step_{step_name}",
            **data,
            basedir=self._stats_dir)
        if len(self.step_stats) == 0 and 'stats' in weight_data and first_step != None:
          # looks like we didn't find any step info but there are model stats. We are probably loading an old model.
          # just load the full model stats as the first step.
          self.step_stats[first_step] = modelstats.ModelStats(
            model_name=f"{self._model.name}{self.run_name}_step_{first_step}",
            **weight_data.get('stats', {}),
            basedir=self._stats_dir)

      if for_train:
        if default_freeze:
          self._model.requires_grad_(False)
        detect_freeze(self._model)
        # Only load the other stuff if they are going to train
        self._optimizers, self._optimizer_args, self._lr_scheduler = self.construct_optimizer(device,
                                                                                              self._model,
                                                                                              missing=missing,
                                                                                              unexpected=unexpected,
                                                                                              load_optimizer=load_optimizer,
                                                                                              optimizer_weights=weight_data.get(
                                                                                                'optimizer',
                                                                                                None),
                                                                                              optimizer_args=weight_data.get(
                                                                                                'optimizer_args', None),
                                                                                              disable_optimizer_warmup=disable_optimizer_warmup,
                                                                                              optimizer_warmup_steps=optimizer_warmup_steps,
                                                                                              optimizer_warmup_fraction=optimizer_warmup_fraction,
                                                                                              **parameters)
        if not isinstance(self._optimizers, list):
          self._optimizers = [self._optimizers]

    else:
      self._model, self._model_args = self._construct_model(device, **parameters)
      self.model_stats = modelstats.ModelStats(model_name=f"{self._model.name}{self.run_name}", basedir=self._stats_dir)
      if for_train:
        if default_freeze:
          self._model.requires_grad_(False)
        detect_freeze(self._model)
        self._optimizers, self._optimizer_args, self._lr_scheduler = self.construct_optimizer(device,
                                                                                              self._model,
                                                                                              **parameters)
        if not isinstance(self._optimizers, list):
          self._optimizers = [self._optimizers]

    self._raw_model = self._model
    if compile_model:
      # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
      self._model = torch.compile(self._model, backend=compile_backend, mode=compile_mode)

    self.scaler = None
    self.amp = NopWith
    if amp:
      if "cuda" in device and not no_grad_scale:
        self.scaler = torch.amp.GradScaler('cuda')
      self.amp = torch.amp.autocast
    self._model.train(for_train)
    self.max_len = self._model.max_len

  def get_model_args(self):
    return self._model_args

  def get_optimizer_args(self):
    return self._optimizer_args

  def save(self, location: str, prod_save=False):
    assert self.is_initialzed(), "Runner not initialized"
    assert self.is_trainable() or prod_save, "Runner not trainable"
    if prod_save:

      checkpoint = {'model': self._raw_model.state_dict(),
                    'model_args': self.get_model_args()}
    else:
      if len(self._optimizers) > 1:
        optimizer_save = [optimizer.state_dict() for optimizer in self._optimizers]
      else:
        optimizer_save = self._optimizers[0].state_dict()

      checkpoint = {'model': self._raw_model.state_dict(),
                    'model_args': self.get_model_args(),
                    'optimizer_args': self.get_optimizer_args(),
                    'optimizer': optimizer_save,
                    'current_step': self.current_step,
                    'stats': self.model_stats.dump_dict(),
                    'step_stats': {stat_name: stat.dump_dict() for stat_name, stat in self.step_stats.items()}}
    if os.path.exists(location):
      copyfile(location, f"{location}.bak")
    torch.save(checkpoint, location)

  def _calculate_stats(self, prompts_data: Sequence[dict], results: Sequence[str]):
    total_words = 0
    total_errors = 0
    total_matches = 0
    for result, prompt_data in zip(results, prompts_data):
      truth = prompt_data['truth']
      result = result.split()
      truth = truth.split()
      total_words += len(truth)
      # The wrong library here can make this the most expensive op in the codebase.
      # This lib is pretty fast though.
      errors, matches = utils.levenshtein_edit_distance(result, truth)

      total_errors += errors
      total_matches += matches
    return total_words, total_errors, total_matches

  def _run_with_truth(self,
                      prompts: Sequence[dict],
                      train: bool,
                      actual_samples_read: int) -> (Sequence[str], torch.Tensor):
    # This will batch to max batch size and pass to the model then re-package the results to return the result.
    # If the passed in batch is more than max_batch_size then gradient accumulation will be used.
    # Tokenization is not done here because the model is the only thing that knows how to do all that.
    assert self.is_initialzed(), "Runner not initialized"
    assert self.is_trainable(), "Runner not trainable"
    mini_batch = []
    batch_results = []
    batch_loss = 0.0
    # Break this into mini-batches that the model can handle
    # amp warns against enclosing the 'backward' but it appears that doing so provides an advantage at least to the optimizer used.
    # leaving it here for now. This should be looked at more in the future.
    with self.amp(device_type=self.device_type):
      for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) >= self.max_batch_size:
          # with self.amp(device_type=self.device_type):
          mini_batch_results, mini_batch_loss = self._model.train_prompts(mini_batch)
          batch_results.extend(mini_batch_results)
          batch_loss = float(mini_batch_loss) + batch_loss
          # mini_batch_loss = mini_batch_loss / (len(mini_batch)/len(prompts))
          mini_batch_loss = mini_batch_loss / len(prompts)
          if train:
            # accumulate the gradients
            if self.scaler is not None:
              self.scaler.scale(mini_batch_loss).backward()
            else:
              mini_batch_loss.backward()

          mini_batch = []
      if len(mini_batch) > 0:
        # with self.amp(device_type=self.device_type):
        mini_batch_results, mini_batch_loss = self._model.train_prompts(mini_batch)
        batch_results.extend(mini_batch_results)
        batch_loss = float(mini_batch_loss) + batch_loss
        # mini_batch_loss = mini_batch_loss / (len(mini_batch)/len(prompts))
        mini_batch_loss = mini_batch_loss / len(prompts)
        if train:
          # accumulate the gradients
          if self.scaler is not None:
            self.scaler.scale(mini_batch_loss).backward()
          else:
            mini_batch_loss.backward()

    # normalize on total samples. Loss should have come back as a sum of the examples normalized on tokens.
    batch_loss = batch_loss / len(prompts)

    # Get basic accuracy stats so we can update the training stats
    tw, te, tm = self._calculate_stats(prompts, batch_results)
    if tw > 0:
      pct_correct = tm / tw
    elif te > 0:
      pct_correct = 0
    else:
      pct_correct = 0
    if train:
      self.model_stats.update_train(len(prompts), pct_correct, float(batch_loss), actual_samples=actual_samples_read)
      self.get_step_stats().update_train(len(prompts), pct_correct, float(batch_loss),
                                         actual_samples=actual_samples_read)
    else:
      self.model_stats.update_validate(len(prompts), pct_correct, float(batch_loss), actual_samples=actual_samples_read)
      self.get_step_stats().update_validate(len(prompts), pct_correct, float(batch_loss),
                                            actual_samples=actual_samples_read)
    return batch_results, batch_loss

  def train(self, prompts: Sequence[dict], actual_samples_read: Optional[int] = None) -> (Sequence[str], torch.Tensor):
    # The assumption is they have sent in a whole batch that they want loss accumulated over
    # But the model may not support the size they send to us.
    # So we will break it into mini batches and do gradient accumulation.
    if not actual_samples_read:
      actual_samples_read = len(prompts)
    for optimizer in self._optimizers:
      optimizer.zero_grad()

    results, current_loss = self._run_with_truth(prompts, True, actual_samples_read)
    if not self.grad_clip is None:
      if not self.scaler is None:
        self.scaler.unscale_(self._optimizers[0])
      torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)

    if self.check_grads:
      for name, param in self._model.named_parameters():
        if param.grad is None:
          print(f"{name} - no gradient found")

    if self.scaler is not None:
      # Scaling only applies to the primary optimizer.
      self.scaler.step(self._optimizers[0])
      self.scaler.update()
    else:
      self._optimizers[0].step()
    for optimizer in self._optimizers[1:]:
      optimizer.step()
    if self._lr_scheduler:
      self._lr_scheduler.step()
    return results, current_loss

  def validate(self, prompts: Sequence[dict], actual_samples_read: Optional[int] = None) -> (
          Sequence[str], torch.Tensor):
    self._model.train(False)
    if not actual_samples_read:
      actual_samples_read = len(prompts)
    results, current_loss = self._run_with_truth(prompts, False, actual_samples_read)
    self._model.train(True)
    return results, current_loss

  def generate(self, prompts: Sequence[str], max_len: Optional[int] = None):
    prompts = [{'prompt': f"{prompt}\n"} for prompt in prompts]
    with self.amp(device_type=self.device_type):
      return self._model.generate_prompts(prompts, max_len)

  def run(self, prompts: Sequence[dict]) -> Sequence[str]:
    # This will batch to max batch size and pass to _run then re-package the results to return the result
    # Tokenization is not done here because the model is the only thing that knows how to do all that.
    batch = []
    results = []
    with torch.no_grad():
      for prompt in prompts:
        batch.append(prompt)
        if len(batch) >= self.max_batch_size:
          results.extend(self._model(batch))
          batch = []
      if len(batch) > 0:
        results.extend(self._model(batch))
    return results

  @abstractmethod
  def _construct_model(self, device, model_weights: dict = None, model_args=None, strict=False, **parameters) -> (
          LMBase, Any):
    pass

  def construct_optimizer(self,
                          device,
                          model: LMBase,
                          missing=None,
                          unexpected=None,
                          load_optimizer=True,
                          optimizer_weights: dict = None,
                          optimizer_args=None,
                          optimizer_warmup_fraction: Optional[float] = None,
                          optimizer_warmup_steps: Optional[int] = None,
                          disable_optimizer_warmup=False,
                          **parameters) -> (Optimizer, Any, Optional[LRScheduler]):
    """Construct one or more optimizers to manage the model. The first optimier is the 'primary' and will have scaling and lr scheduling applied if availale.
    secondary optimizers are needed if training on different types of devices (CPU + GPU) with amp. Multiple optimizers is rarely needed.
    To fix parameters to a secondary optizer just tag them with 'blah.actual_parameter.secondary_optimizer = True' when they are constructed.

    :param device:
    :param model:
    :param missing:
    :param unexpected:
    :param load_optimizer:
    :param optimizer_weights:
    :param optimizer_args:
    :param optimizer_warmup_fraction:
    :param optimizer_warmup_steps:
    :param disable_optimizer_warmup:
    :param parameters:
    :return:
    """

    if optimizer_args is None:
      optimizer_args = dict()
    lr = parameters.get('lr', optimizer_args.get('lr', DEFAULT_LR))
    weight_decay = parameters.get('weight_decay', optimizer_args.get('weight_decay', DEFAULT_WEIGHT_DECAY))
    primary_weights = []
    secondary_weights = []
    # Detect primary and secondary optimizer targets.
    for parameter in model.parameters():
      if hasattr(parameter, "secondary_optimizer") and parameter.secondary_optimizer:
        secondary_weights.append(parameter)
      else:
        primary_weights.append(parameter)
    optimizers = [torch.optim.Adagrad(primary_weights, lr=lr, weight_decay=weight_decay)]
    if len(secondary_weights) > 0:
      optimizers.append(torch.optim.Adagrad(secondary_weights, lr=lr, weight_decay=weight_decay))
    lr_scheduler = None
    if optimizer_weights is not None and not isinstance(optimizer_weights, list):
      optimizer_weights = [optimizer_weights]
    if optimizer_weights is not None and load_optimizer and len(unexpected) == 0 and len(missing) == 0 and len(
            optimizer_weights) == len(optimizers):
      # if optimizer_weights is not None and load_optimizer and len(missing) == 0 and len(unexpected) == 0:
      # only load old optimizers if the model parameters haven't changed.
      optimizer_loaded = True
      for optimizer, weights in zip(optimizers, optimizer_weights):
        for pg in weights['param_groups']:
          if 'lr' in pg:
            pg['lr'] = lr
          if 'weight_decay' in pg:
            pg['weight_decay'] = weight_decay
        if load_optimizer:
          try:
            optimizer.load_state_dict(weights)
          except ValueError:
            optimizer_loaded = False
            print(
              "Unable to load optimizer. Probably a new parameter that is throwing things off. (This optimizer doesn't belong to this model)")
        else:
          optimizer_loaded = False
    else:
      optimizer_loaded = False
    create_warmup_scheduler = not disable_optimizer_warmup and optimizer_weights is not None and not optimizer_loaded and len(
      optimizers) == 1
    if create_warmup_scheduler:
      print("Using warmup scheduler.")
      lr_scheduler = OptimizerWarmupLRScheduler(optimizers[0],
                                                steps=optimizer_warmup_steps,
                                                initial_fraction=optimizer_warmup_fraction)
    optimizer_args['lr'] = lr
    optimizer_args['weight_decay'] = weight_decay
    if len(optimizers) > 1:
      return optimizers, optimizer_args, lr_scheduler
    return optimizers[0], optimizer_args, lr_scheduler
