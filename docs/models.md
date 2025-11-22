# Models and Runners

This document explains the model and runner abstractions that form the core of LMPlay.

## Overview

LMPlay separates **model definition** from **training infrastructure** through two key classes:

1. **Models** (`LMBase` subclasses): Define the neural network architecture
2. **Runners** (`LMRunnerBase` subclasses): Handle training, optimization, and checkpointing

This separation allows the same training code to work with different models.

## Model Base Classes

### MBase

The foundational model class (`lmplay/base/base_model.py:74-369`).

**Responsibilities**:
- Tokenization (converting text to token IDs)
- Training interface (computing loss for prompt/truth pairs)
- Generation interface (autoregressive text generation)
- Device management
- Parameter counting

**Key Methods**:

```python
def _tokenize_str(sample: dict, device, trim=True) -> (Tensor, int):
    """Tokenize a single sample.

    Returns: (tokens tensor, prediction_start index)
    """

def _tokenize_batch(batch: Sequence[dict], dont_pad=False) -> (Tensor, List[int]):
    """Tokenize and pad a batch of samples."""

def train_prompts(prompts: Sequence[dict], include_prompts=True) -> (List[str], Tensor, int):
    """Compute training loss on prompt/truth pairs.

    Returns: (predictions, loss, token_count)
    """

def generate_prompts(prompts: Sequence[dict], max_len: Optional[int]) -> List[str]:
    """Generate completions for prompts (no gradients).

    Uses key-value caching for efficient generation.
    """

def parameter_count() -> int:
    """Total number of parameters in the model."""
```

### LMBase

Specialized for language modeling (`lmplay/base/base_model.py:371-389`).

**Key difference**: Defines the forward signature for autoregressive models:

```python
@abstractmethod
def forward(self, x: torch.Tensor, cache: Optional[List] = None) -> torch.Tensor:
    """Forward pass.

    Args:
        x: Input token indices, shape (batch_size, seq_len)
        cache: Optional key-value cache from previous generation steps

    Returns:
        Logits of shape (batch_size, seq_len, vocab_size) for training
        or (batch_size, 1, vocab_size) for generation (with cache)
    """
```

**How to subclass**:

```python
from lmplay.base.base_model import LMBase
from torch import nn

class MyModel(LMBase):
    def __init__(self, max_len=1024, embed_dim=768, **kwargs):
        super().__init__(
            name="my_model",
            max_len=max_len,
            embed_dim=embed_dim,
            **kwargs
        )
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.embedding = nn.Embedding(self.tokenizer.n_vocab, embed_dim)
        # ... rest of your architecture

    def forward(self, x, cache=None):
        # Your forward pass implementation
        # Return logits (and optionally updated cache for generation)
        pass
```

## Runner Base Classes

### LMRunnerBase

The abstract runner class (`lmplay/base/base_model.py:428-1045`).

**Responsibilities**:
- Creating and initializing models
- Managing optimizers and learning rate schedulers
- Implementing training loops with gradient accumulation
- Checkpointing and resume functionality
- Statistics tracking

**Key Methods**:

```python
def initialize(self,
               device: str,
               locations: Optional[Union[Sequence[str], str]] = None,
               for_train: bool = True,
               load_optimizer: bool = True,
               num_blocks: int = 6,
               **parameters):
    """Initialize the runner with a model on a device.

    Args:
        device: 'cuda', 'cpu', 'mps', etc.
        locations: Path(s) to checkpoint file(s) to load
        for_train: If True, initialize optimizer (for training).
                   If False, load in eval mode (for inference).
        load_optimizer: Whether to load optimizer state from checkpoint
        num_blocks: Number of layers (passed to model)
        **parameters: Additional parameters passed to model
    """

def train(self,
          prompts: Sequence[dict],
          actual_samples_read: Optional[int] = None) -> (List[str], Tensor, int):
    """Execute one training step.

    Args:
        prompts: List of dicts with 'prompt' and 'truth' keys
        actual_samples_read: For statistics (usually len(prompts))

    Returns:
        (predictions, loss, token_count)
    """

def validate(self,
             prompts: Sequence[dict],
             actual_samples_read: Optional[int] = None) -> (List[str], Tensor, int):
    """Execute validation step (no gradients)."""

def generate(self,
             prompts: Sequence[str],
             max_len: Optional[int] = None) -> List[str]:
    """Generate text completions."""

def save(self, location: str, prod_save: bool = False):
    """Save model checkpoint.

    If prod_save=True: Only save model weights
    If prod_save=False: Save model, optimizer, and statistics
    """

@abstractmethod
def _construct_model(self,
                     device,
                     model_weights: dict = None,
                     model_args: dict = None,
                     strict: bool = False,
                     **parameters) -> (LMBase, dict):
    """Construct model instance. Must be implemented by subclasses."""

def construct_optimizer(self,
                        device,
                        model: LMBase,
                        missing: List = None,
                        unexpected: List = None,
                        load_optimizer: bool = True,
                        optimizer_weights: dict = None,
                        **parameters) -> (Optimizer, dict, Optional[LRScheduler]):
    """Create optimizers (supports primary + secondary for multi-device AMP)."""
```

### BasicModelRunner

Simple runner implementation (`lmplay/base/base_model.py:1048-1100`).

The most commonly used runner for straightforward models.

```python
class BasicModelRunner(LMRunnerBase):
    def __init__(self,
                 model_class,
                 max_batch_size: int = 25,
                 overrides: dict = None,
                 **kwargs):
        """
        Args:
            model_class: Your model class (subclass of LMBase)
            max_batch_size: GPU batch size for gradient accumulation
            overrides: Parameter overrides (e.g., for model variants)
        """
        super().__init__(max_batch_size=max_batch_size, **kwargs)
        self.model_class = model_class
        self.overrides = overrides or {}

    def _construct_model(self, device, model_weights=None, model_args=None,
                        strict=False, **parameters):
        # Merge parameters and overrides
        if model_args is None:
            model_args = {}

        model_args.update(parameters)
        model_args.update(self.overrides)

        # Instantiate
        model = self.model_class(**model_args)

        # Load weights if provided
        if model_weights is not None:
            missing, unexpected = model.load_state_dict(model_weights, strict=strict)
            model.to(device)
            return model, model.init_kwargs, missing, unexpected

        model.to(device)
        return model, model.init_kwargs
```

## Training Loop Details

### Gradient Accumulation

LMPlay automatically handles gradient accumulation based on batch sizes:

```python
# User specifies:
batch_size = 50           # Desired effective batch size
mini_batch_size = 4       # GPU memory constraint

# Framework does:
# 1. Break the 50 samples into batches of 4
# 2. For each mini-batch:
#    - Compute loss and scale by (4 / 50)
#    - Call loss.backward() to accumulate gradients
# 3. After processing all mini-batches (50 samples):
#    - Call optimizer.step()
# 4. Normalize accumulated loss by total tokens
```

This is handled in `LMRunnerBase._run_with_truth()` (line 739-840).

### Checkpoint Format

Checkpoints are PyTorch `.pt` files containing:

```python
{
    'model': model_state_dict,              # Model weights
    'model_args': {...},                    # Model init parameters
    'optimizer': optimizer_state_dict,      # Optimizer state
    'optimizer_args': {...},                # Optimizer configuration
    'current_step': 'stage_name',           # For multi-stage training
    'stats': {...},                         # Training statistics (loss, accuracy)
    'step_stats': {                         # Per-stage statistics
        'stage_1': {...},
        'stage_2': {...},
    }
}
```

Resume works by:
1. Loading all state from checkpoint
2. Continuing from `current_step`
3. Fast-forwarding dataset to previous position via `fast_forward` parameter

### Optimizer Management

The framework uses **Adagrad** optimizer by default:
- Learning rate: 6e-4
- Weight decay: 0.0

**Secondary optimizer support** for multi-device training (CPU + GPU with AMP):
- Parameters can be tagged with `param.secondary_optimizer = True`
- Each device type gets its own optimizer
- Useful when AMP can't work on CPU layers

## Custom Runners

For complex scenarios, subclass `LMRunnerBase` directly:

```python
class CustomRunner(LMRunnerBase):
    def _construct_model(self, device, model_weights=None, **parameters):
        # Custom model construction logic
        model = MyModel(**parameters)
        if model_weights is not None:
            model.load_state_dict(model_weights)
        model.to(device)
        return model, model.init_kwargs

    def construct_optimizer(self, device, model, **parameters):
        # Custom optimizer setup (e.g., different optimizer, custom schedule)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = ... # custom scheduler
        return optimizer, {}, scheduler
```

## Device Management

Models are expected to have a cached `device` property:

```python
@property
def device(self):
    """Get model device from fc layer weights."""
    return self.fc.weight.device
```

This allows the framework to determine device without external state.

## Tokenization

All models use **TikToken GPT-2 encoding**:

```python
self.tokenizer = tiktoken.get_encoding("gpt2")
```

Token indices:
- 0-50256: Regular tokens
- 50256: `<|endoftext|>` (EOT token)

Tokenization ensures each sample has:
- `prompt`: Initial text
- `truth`: Text to predict

Loss is computed only on `truth` tokens.

## Type Hints and Documentation

All public methods have:
- Complete type hints
- Docstrings explaining parameters and returns
- Clear error messages

Follow this pattern when extending LMPlay.