# Training System

## Overview

The training system handles:

1. **Dataset loading and batching** - HuggingFace datasets with intelligent text continuation
2. **Multi-stage training** - Curriculum learning via training plans
3. **Gradient accumulation** - Handling GPU memory constraints
4. **Checkpointing and resume** - Graceful handling of interruptions
5. **Statistics tracking** - Loss, accuracy, tokens per stage
6. **Flexible optimizers** - Support for multiple optimizer types with intelligent weight decay

## Training Entry Point

```bash
lmp_trainer [options]
```

Location: `lmplay/train/__main__.py`

## Configuration

### Core Parameters

```bash
# Model selection
--exp gpt2ish              # Experiment name
--num-blocks 6             # Number of transformer layers
--embed-dim 768            # Embedding dimension
--num-heads 12             # Number of attention heads
--context_len 1024         # Maximum sequence length (optional)

# Device and optimization
--device cuda              # 'cuda', 'cpu', or 'mps'
--amp                      # Enable Automatic Mixed Precision
--compile-model            # Use torch.compile (experimental)

# Batch sizes
--batch-size 50            # Effective batch size (logical)
--mini-batch-size 4        # GPU batch size (memory constraint)
--validation-batch-size 4  # Validation batch size

# Training schedule
--lr 0.0006                # Learning rate
--grad-clip 0.1            # Gradient clipping (optional)
--training-plan default    # Multi-stage training plan

# Checkpointing
--model my_model.lmp       # Model file to load/save
--initial-model init.lmp   # Fallback if --model not found
--save-interval 10000      # Save every N samples
--validation-interval 100  # Validate every N samples
--reset-history            # Clear previous training stats

# Optimizer and regularization
--optimizer adamw          # Optimizer: adamw (default), adam, adagrad, sgd, rmsprop
--weight-decay 0.01        # L2 regularization (None = optimizer default)

# Advanced
--optimizer-warmup-fraction 0.1      # LR warmup start
--optimizer-warmup-steps 40          # Warmup duration
--disable-optimizer-warmup           # Skip warmup
--default-freeze           # Freeze all params by default
--ignore-optimizer         # Don't load optimizer state
--check-grads              # Print params with no gradients
```

## Datasets

### HuggingFace Datasets

Default: Wikipedia (English and Spanish)

**Loading**:
```python
dataset = datasets.load_dataset('wikipedia', '20220601.en')
```

**Supported formats**:
- `wikipedia`: Multi-language Wikipedia
- `wikitext`: WikiText versions
- Any HuggingFace dataset with a 'text' field

### Batching and Text Continuation

**Problem**: Text samples have variable length. How to batch them?

**Solution**: Intelligent continuation at sentence boundaries.

```python
# Example: 1000-word article split into 3 training samples
Article (1000 words)
  ├─ Sample 1: Prompt: "Title..." → Truth: "First 300 words..."
  ├─ Sample 2: Prompt: "...end of sample 1. " → Truth: "Next 300 words..."
  └─ Sample 3: Prompt: "...end of sample 2. " → Truth: "Final 400 words..."
```

**Key properties**:
- Continuations never cross batch boundaries (prevents cheating during training)
- Continuation points use sentence boundaries (text boundary respect)
- Samples preserve chronological order (for repeatable runs)
- Token estimation uses 1.5 tokens/word heuristic

**Code**: `lmplay/train/datasets/utils.py:124-205`

## Training Plans

Multi-stage training allows curriculum learning or progressive training.

### Built-in Plans

```bash
# Check plan_configs.py for all plans
--training-plan default    # Single Wikipedia stage
--training-plan wikitext3  # Different text source
```

### Custom Plans

Create a JSON file:

```json
{
  "pretraining": {
    "dataset": "wikipedia",
    "epochs": 3
  },
  "finetune": {
    "dataset": "wikitext",
    "epochs": 1
  }
}
```

Then use:
```bash
lmp_trainer --training-plan /path/to/plan.json
```

**Plan format**:
- Key = stage name (used for statistics)
- `dataset`: HuggingFace dataset name
- `epochs`: Number of passes over dataset (can be fractional)

### Resume and Fast-Forward

When resuming from checkpoint:
1. Load `current_step` from checkpoint
2. Start from that stage
3. Use `fast_forward` parameter to skip already-seen samples
4. This is detected automatically from `model_stats.total_train_samples`

## Training Loop

```
Input: Training plan, runner initialized with model/optimizer

For each stage in plan:
  │
  ├─ Load dataset from HuggingFace
  ├─ Create batcher with intelligent continuation
  │
  └─ For each batch:
     │
     ├─ For each mini-batch (gradient accumulation):
     │  │
     │  ├─ Forward pass: loss = model.train_prompts(mini_batch)
     │  ├─ Scale loss by (mini_batch_size / batch_size)
     │  └─ loss.backward() accumulate gradients
     │
     ├─ Optimizer.step() update weights
     ├─ Update learning rate schedule (if warmup)
     │
     ├─ Check if validation due:
     │  └─ Run validation batch, print prediction example
     │
     └─ Check if save due:
        └─ Save checkpoint with model, optimizer, stats

End of plan: Final checkpoint saved
```

**Code**: `lmplay/train/__main__.py:270-334`

## Gradient Accumulation

### Problem

Your effective batch size (50) exceeds GPU memory (4).

### Solution

```
Effective batch = 50 samples
GPU capacity = 4 samples

Framework does:
├─ Mini-batch 1 (samples 0-3):   loss1.backward()  [accum]
├─ Mini-batch 2 (samples 4-7):   loss2.backward()  [accum]
├─ Mini-batch 3 (samples 8-11):  loss3.backward()  [accum]
├─ ... (13 mini-batches total)
└─ Mini-batch 13 (samples 48-50): loss13.backward()  [accum]
   Then: optimizer.step()        [update]
```

Each loss is scaled by `mini_batch_size / batch_size` before backward() to maintain correct gradient magnitude.

### Configuration

```bash
lmp_trainer --batch-size 50 --mini-batch-size 4
# Framework automatically handles splitting
```

## Optimizers and Weight Decay

The framework supports multiple optimizers with intelligent weight decay handling.

### Available Optimizers

```bash
--optimizer adamw       # Default - Best for most tasks, includes weight decay
--optimizer adam        # Adaptive learning rates, no weight decay by default
--optimizer adagrad     # Sparse gradient optimization
--optimizer sgd         # Classical stochastic gradient descent with momentum
--optimizer rmsprop     # Root mean square propagation
```

**Default**: AdamW with weight decay of 0.01

### Weight Decay Defaults

Each optimizer has sensible defaults:

| Optimizer | Default Weight Decay |
|-----------|----------------------|
| **AdamW** | 0.01 (recommended for regularization) |
| Adam      | 0.0 (no regularization) |
| Adagrad   | 0.0 (no regularization) |
| SGD       | 0.0 (no regularization) |
| RMSprop   | 0.0 (no regularization) |

### Custom Weight Decay

```bash
# Override weight decay
lmp_trainer --optimizer adamw --weight-decay 0.001

# Disable weight decay
lmp_trainer --optimizer adamw --weight-decay 0

# Use optimizer default (no --weight-decay flag)
lmp_trainer --optimizer adamw  # Uses 0.01
```

### Smart Weight Decay: Parameter Exclusions

By default, certain parameter types are **excluded from weight decay**:

- **Bias parameters**: `.bias`, `_bias`
- **LayerNorm weights**: `.ln`, `_ln` (layer normalization)
- **Embeddings**: `embed` (token embeddings, positional embeddings, etc.)

This is the standard practice in transformer training - we don't want to regularize normalization and embedding layers.

**Example**: With weight_decay=0.01:
```
Regular weights (attention, MLP):    weight_decay=0.01  ✓ regularized
Bias terms:                          weight_decay=0.0   ✓ excluded
LayerNorm weights:                   weight_decay=0.0   ✓ excluded
Embeddings:                          weight_decay=0.0   ✓ excluded
```

### Excluding Custom Parameters from Weight Decay

In your model code, you can mark specific parameters to skip weight decay:

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Regular weight - will have weight decay applied
        self.fc = nn.Linear(10, 10)

        # Sacrificial parameter - exclude from decay
        self.sacrificial = nn.Parameter(torch.randn(10, 10))
        self.sacrificial.skip_weight_decay = True
        self.register_parameter('sacrificial_param', self.sacrificial)

        # Another example - custom scaling factor
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.scale.skip_weight_decay = True
```

### Usage Examples

**Train with AdamW (recommended default)**:
```bash
lmp_trainer --amp --device cuda --exp gpt2ish
# Uses: AdamW optimizer, weight_decay=0.01
# Excludes: bias, layer norm, embeddings from decay
```

**Use SGD for comparison**:
```bash
lmp_trainer --amp --device cuda --optimizer sgd --lr 0.1
# Uses: SGD with momentum, no weight decay
```

**Strong regularization with AdamW**:
```bash
lmp_trainer --amp --device cuda --optimizer adamw --weight-decay 0.1
# Uses: AdamW with 0.1 weight decay (strong)
```

**No regularization**:
```bash
lmp_trainer --amp --device cuda --optimizer adam --weight-decay 0
# Uses: Adam with NO weight decay
```

## Validation

### Running Validation

```bash
--validation-interval 100      # Validate every 100 training samples
--validation-batch-size 4      # Size of validation batch
```

### What Gets Validated

```
Validation batch (4 samples):
├─ Forward pass (no gradients)
├─ Compute loss on truth tokens
├─ Track accuracy (edit distance)
└─ Print one example:
   Input:  "Once upon a time..."
   Truth:  "there lived a..."
   Pred:   "the king ruled..."  (model's prediction)
```

### Statistics Tracked

Per batch:
- `loss`: Cross-entropy loss
- `accuracy`: Word-level accuracy (from edit distance)
- `tokens`: Total tokens in batch

Per step:
- Running averages of above
- Saved to JSON file for plotting

## Checkpointing

### Automatic Saves

```bash
--save-interval 10000    # Save after every 10,000 samples
```

Creates: `model_name.stage_name.lmp`

### Manual Inspection

```python
import torch

checkpoint = torch.load('model.lmp', weights_only=False)

# Checkpoint contents (organized by component):
print(checkpoint.keys())
# dict_keys(['curriculum', 'model', 'runner', 'optimizers', 'lr_schedulers'])

# Each component has state_args and state:
print(checkpoint['model'].keys())
# dict_keys(['state_args', 'state'])

# Access model weights and construction args
model_weights = checkpoint['model']['state']  # Model state_dict
model_args = checkpoint['model']['state_args']  # Model construction parameters

# Access training statistics from runner component
runner_state = checkpoint['runner']['state']
stats = runner_state['stats']  # Overall training statistics
print(f"Total samples trained: {stats['total_train_samples']}")
print(f"Total loss: {stats['train_loss_total']}")

# Access per-stage statistics
step_stats = runner_state['step_stats']  # Dict of statistics per training stage
for stage_name, stage_stat in step_stats.items():
    print(f"{stage_name}: {stage_stat['total_train_samples']} samples")
```

### Resume Training

```bash
# Automatically resumes from model_name.lmp if it exists
lmp_trainer --model model_name.lmp

# Or force fresh training (ignores old checkpoint)
lmp_trainer --model model_name.lmp --reset-history
```

## Statistics

### Tracking

Statistics are tracked in `lmplay/stats/modelstats.py`.

Each runner has:
- `model_stats`: Overall statistics
- `step_stats`: Per-stage statistics (for multi-stage training)

### Output

Saved to `out_gpt/` (or `LMP_DATASETS` env var):

```
out_gpt/
├── model_name_train_stats.json
├── model_name_train_stats.png
├── model_name_validate_stats.json
├── model_name_validate_stats.png
├── model_name_step_stage1_train_stats.json
├── model_name_step_stage1_validate_stats.json
├── ...
```

### Plotting

```bash
lmp_plotstats      # Plots all stats, creates PNG files
```

Creates both:
- **Regular plots**: Absolute metrics
- **Diff plots**: Normalized to longest run (easier comparison)

## Error Handling

### Graceful Interruption

```bash
lmp_trainer --device cuda
# [Press Ctrl+C]

# Framework catches KeyboardInterrupt:
# 1. Saves current step (*.stage_name.lmp)
# 2. Updates statistics
# 3. Exits cleanly

# Next run resumes automatically
lmp_trainer --device cuda  # Continues from saved point
```

### Issues

If training fails:
1. Check the full stack trace (never caught silently)
2. Model checkpoint saved at last interval (check `--save-interval`)
3. `out_gpt/` directory may have partial stats

## Performance Tuning

### Slow Training?

1. **Increase batch size if GPU memory allows**:
   ```bash
   --mini-batch-size 8  # Try doubling
   ```

2. **Use AMP (Automatic Mixed Precision)**:
   ```bash
   --amp  # Much faster on CUDA
   ```

3. **Use torch.compile (experimental)**:
   ```bash
   --compile-model --compile-mode reduce-overhead
   ```

4. **Adjust validation frequency**:
   ```bash
   --validation-interval 1000  # Validate less often
   ```

### Memory Issues?

1. **Reduce mini-batch-size**:
   ```bash
   --mini-batch-size 2
   ```

2. **Reduce validation-batch-size**:
   ```bash
   --validation-batch-size 2
   ```

3. **Reduce context-len** (if too large):
   ```bash
   --context_len 512
   ```

## Troubleshooting

### "model shock" after loading checkpoint

You see: Initial normal loss → sudden spike → recovery

**Solution**: Use optimizer warmup
```bash
--optimizer-warmup-fraction 0.1     # Start at 10% LR
--optimizer-warmup-steps 40         # Warm up over 40 batches
```

### Different results between runs

Likely caused by:
- Different hardware (CUDA vs CPU, different GPU)
- Non-deterministic operations (PyTorch's cublas)
- Dataset order differs if not using deterministic seeding

**Note**: LMPlay is designed for research and experimentation, not reproducibility.

### Model not improving

Check:
1. Is `--validation-interval` showing decreasing loss?
2. Try larger `--batch-size` for more stable gradients
3. Check `--describe` to verify architecture
4. Try `--check-grads` to find frozen parameters

## Dataset Caching

HuggingFace datasets are cached locally:

```bash
# Check cache location
du -sh ~/.cache/huggingface/datasets/

# To save datasets for transfer to another machine
lmp_trainer --save-datasets

# Downloads and saves to out_gpt/datasets/
# Then copy to another machine and point to it
```