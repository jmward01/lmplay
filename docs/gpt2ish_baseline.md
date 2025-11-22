# GPT2ish Baseline Model

The GPT2ish baseline is the reference implementation for LMPlay. It serves as:

1. **Template for new experiments**: Copy this to create variants
2. **Reference for expected behavior**: Shows how models should be structured
3. **Performance baseline**: For comparing new architectures

**Location**: `lmplay/exp/gpt2ish/`

## Model Architecture

### Overview

```
Input Tokens
    ↓
Token Embedding (vocab_size → 768)
    ↓
Positional Embedding (learnable, 1 × 1024 × 768)
    ↓
Dropout (0.1)
    ↓
Transformer Block × 6
    ├─ Multi-head Attention (12 heads)
    ├─ Feed-Forward (768 → 3072 → 768)
    └─ Residual connections
    ↓
Layer Normalization
    ↓
Output Projection (768 → vocab_size)
    ↓
Logits
```

### Design Decisions

1. **GPT-2 style, not GPT-2 reference**: This is a simplified implementation, not based on OpenAI's official code
2. **Pre-norm architecture**: Layer norm before attention and feed-forward (more stable)
3. **Learnable positional embeddings**: Simpler than rotary or other schemes
4. **Standard attention**: Causal masking (no looking ahead)
5. **12-head attention**: Balanced between expressiveness and performance

## Implementation Details

### Model File: `model.py`

```python
class GPT2(LMBase):
    """GPT2-like transformer model."""

    def __init__(self,
                 max_len=1024,
                 num_heads=12,
                 num_blocks=6,
                 embed_dim=768,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 embed_dropout=0.1,
                 version="gpt2ish",
                 **ignore):
```

**Parameters**:
- `max_len`: Sequence length (default 1024 - don't change without reason)
- `num_heads`: Attention heads (must divide `embed_dim`)
- `num_blocks`: Number of transformer layers
- `embed_dim`: Embedding dimension
- `attn_dropout`: Dropout in attention heads
- `ff_dropout`: Dropout in feed-forward network
- `embed_dropout`: Dropout after embeddings
- `version`: Name prefix for checkpoints
- `**ignore`: Silently ignore unknown parameters

**Key methods**:

```python
def forward(self, x: torch.Tensor, cache: Optional[List] = None):
    """Forward pass for training or generation.

    Training:
        x: shape (batch_size, seq_len)
        cache: None
        → returns logits of shape (batch_size, seq_len, vocab_size)

    Generation:
        x: shape (batch_size, 1) - single new token
        cache: list of previous KV states
        → returns logits of shape (batch_size, 1, vocab_size), updated cache
    """
```

### Runner File: `versions.py`

```python
@expose_runner('gpt2ish',
               description='Reference model. Loosely based on GPT-2.')
def runner(*args, **kwargs):
    return BasicModelRunner(GPT2,
                           *args,
                           overrides=dict(),
                           **kwargs)
```

**Additional versions**:
- `gpt2ish_med`: 24-layer, 1024 embedding, 16 heads (medium model)
- `gpt2ish_large`: 36-layer, 1280 embedding, 20 heads (large model)
- `test_runner`: Tiny model (2 heads, 64 dim) for quick testing
- `s_gpt2ish`: Small and deep (18 layers, 288 embedding, 6 heads)

## Using as a Template

### Step 1: Copy the Experiment

```bash
cp -r lmplay/exp/gpt2ish lmplay/exp/my_experiment
```

### Step 2: Modify `model.py`

```python
# lmplay/exp/my_experiment/model.py
from lmplay.base.base_model import LMBase
from lmplay.modules import Block
import torch.nn as nn
import tiktoken

class MyModel(LMBase):
    def __init__(self,
                 max_len=1024,
                 num_heads=12,
                 num_blocks=6,
                 embed_dim=768,
                 # Your custom parameters
                 my_custom_param=True,
                 **ignore):
        super().__init__(to_name('my_experiment',
                                 num_blocks=num_blocks,
                                 max_len=max_len),
                        max_len=max_len,
                        num_heads=num_heads,
                        num_blocks=num_blocks,
                        embed_dim=embed_dim,
                        my_custom_param=my_custom_param)

        self.tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = self.tokenizer.n_vocab

        # YOUR ARCHITECTURE HERE
        # Copy from GPT2 and make changes
        # ...

    def forward(self, x, cache=None):
        # YOUR IMPLEMENTATION HERE
        pass
```

### Step 3: Update `versions.py`

```python
# lmplay/exp/my_experiment/versions.py
from lmplay.base.base_model import BasicModelRunner
from .model import MyModel
from lmplay.base.runner_list import expose_runner

@expose_runner('my_experiment',
               description='My custom experiment description')
def runner(*args, **kwargs):
    return BasicModelRunner(MyModel,
                           *args,
                           overrides=dict(),
                           **kwargs)

# Optional: Add other versions
@expose_runner('my_experiment_large',
               description='Larger version of my experiment')
def runner_large(*args, **kwargs):
    return BasicModelRunner(MyModel,
                           *args,
                           overrides=dict(num_blocks=24,
                                         embed_dim=1024,
                                         num_heads=16),
                           **kwargs)
```

### Step 4: Register in `lmplay/__init__.py`

```python
# Add at the end of lmplay/__init__.py
import lmplay.exp.my_experiment.versions
```

### Step 5: Test

```bash
# Check it's registered
lmp_trainer --exp list | grep my_experiment

# Describe it
lmp_trainer --exp my_experiment --describe

# Train it
lmp_trainer --exp my_experiment --device cpu --mini-batch-size 1
```

## Common Modifications

### Change Architecture

```python
# Use 8 attention heads instead of 12
self.num_heads = 8

# Use 2048 hidden size in FFN instead of 4*embed_dim
# Modify Block construction or create custom block
```

### Add Custom Layers

```python
# Before blocks
self.special_layer = MySpecialLayer(embed_dim)

# In forward pass
x = self.special_layer(x)
for block in self.blocks:
    x = block(x, cache=...)
```

### Conditional Computation

```python
# Example: Route through different paths
if self.training:
    x = self.blocks_train(x)
else:
    x = self.blocks_inference(x, cache)
```

## Training with Baseline

### Quick Test

```bash
lmp_trainer --exp gpt2ish --device cpu --mini-batch-size 1 \
  --validation-interval 10 --save-interval 1000000
```

### Standard Training

```bash
lmp_trainer --exp gpt2ish --device cuda --amp \
  --batch-size 50 --mini-batch-size 4 \
  --validation-interval 100 --save-interval 10000
```

### Multi-stage Training

```bash
# Create a training plan
cat > my_plan.json << 'EOF'
{
  "stage1_pretrain": {
    "dataset": "wikipedia",
    "epochs": 1
  },
  "stage2_finetune": {
    "dataset": "wikitext",
    "epochs": 1
  }
}
EOF

lmp_trainer --exp gpt2ish --device cuda --amp \
  --training-plan my_plan.json
```

## Expected Performance

### Loss Curves

On Wikipedia (default):
- Initial loss: ~4.5 (vocab size ≈ 50k)
- After 10k samples: ~4.0-4.2
- After 100k samples: ~3.5-3.8
- Converges slowly - this is expected on small models

### Generation Quality

- 6 layers: Mediocre, mostly learns local patterns
- 12 layers: Better, generates coherent sentences
- 24+ layers: Good, generates longer coherent text

**Note**: These are loose estimates - actual values depend on dataset, learning rate, and other factors.

## Modifying Baseline Versions

### Create a Smaller Model

```python
@expose_runner('gpt2ish_small',
               description='2-layer, 256-dim model for fast testing')
def runner_small(*args, **kwargs):
    return BasicModelRunner(GPT2,
                           *args,
                           overrides=dict(num_blocks=2,
                                         embed_dim=256,
                                         num_heads=4),
                           **kwargs)
```

### Create a Non-Causal Model

```python
# Copy GPT2 to causal_gpt2.py
# Modify attention mask to allow all positions to attend to all positions
# Create new runner for it
```

### Add Experiment-Specific Features

```python
@expose_runner('gpt2ish_with_aux_loss',
               description='GPT2ish with auxiliary loss head')
def runner_aux(*args, **kwargs):
    return BasicModelRunner(GPT2WithAuxLoss,  # Custom class
                           *args,
                           overrides=dict(),
                           **kwargs)
```

## Debugging Issues

### Model doesn't train

1. Check `--describe` output matches what you expect
2. Check `--check-grads` for frozen parameters
3. Start with tiny model: `--num-blocks 2 --embed-dim 128`
4. Check learning rate: try `--lr 0.001`

### Generation produces noise

1. Model probably untrained (validate loss first)
2. Reduce generation temperature (in generate/__main__.py)
3. Check KV-cache implementation if implementing custom block

### Memory issues

1. Reduce `mini_batch_size`
2. Reduce `max_len`
3. Reduce `embed_dim`
4. Use CPU for debugging

## File Checklist for New Experiment

When creating from baseline, ensure you have:

- [ ] `lmplay/exp/your_experiment/__init__.py` (can be empty)
- [ ] `lmplay/exp/your_experiment/model.py` (your model class)
- [ ] `lmplay/exp/your_experiment/versions.py` (runner(s))
- [ ] Updated `lmplay/__init__.py` (import statement)
- [ ] Test runs without errors
- [ ] README or docstrings explaining your changes

## Next Steps

After getting the baseline working:

1. **Read [Architecture](./architecture.md)** for system design understanding
2. **Read [Models & Runners](./models.md)** for extending base classes
3. **Experiment**: Modify architecture, test, compare results
4. **Iterate**: Update, train again, compare metrics

The baseline is your starting point. Explore confidently - failures are part of research!