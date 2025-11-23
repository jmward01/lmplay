# LMPlay Architecture

## High-Level Overview

LMPlay is structured around two key patterns:

1. **The Runner Pattern**: Models are wrapped in "runners" that handle training infrastructure
2. **The Experiment Pattern**: New models are created by copying from baselines and modifying

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Infrastructure                  │
│  (Training Loop, Datasets, Optimization, Checkpointing)     │
│  lmplay/train, lmplay/stats, lmplay/base/base_model.py     │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                 ┌──────────┴──────────┐
                 │                     │
         ┌───────▼──────┐      ┌───────▼──────┐
         │   Runners    │      │   Runners    │
         │ (LMRunnerBase│      │ (LMRunnerBase│
         │  + subclass) │      │  + subclass) │
         └───────┬──────┘      └───────┬──────┘
                 │                     │
         ┌───────▼──────┐      ┌───────▼──────┐
         │   Models     │      │   Models     │
         │  (LMBase     │      │  (LMBase     │
         │   subclass)  │      │   subclass)  │
         └──────────────┘      └──────────────┘

         GPT2ish Baseline     Your Experiment
         lmplay/exp/gpt2ish   lmplay/exp/your_exp
```

## Core Components

### 1. Model Hierarchy

```
nn.Module (PyTorch)
    └── MBase (lmplay/base/base_model.py)
        └── LMBase (lmplay/base/base_model.py)
            └── Concrete Model (e.g., GPT2, GPT2ish)
```

**MBase** provides:
- Tokenization (`_tokenize_str`, `_tokenize_batch`)
- Training interface (`train_prompts`)
- Generation interface (`generate_prompts`)
- Device management
- Parameter counting

**LMBase** specializes for language modeling:
- Forward signature: `forward(x: Tensor, cache: Optional[List]) -> Tensor`
- Supports key-value caching for efficient generation

### 2. Runner Hierarchy and Component Architecture

```
LMRunnerBase (ABC, lmplay/base/base_runner.py)
    └── BasicModelRunner (for standard models, lmplay/base/runners.py)
        └── Custom Runners (optional, for special cases)
```

**LMRunnerBase** uses a **component system** that cleanly separates concerns. Five core components manage different aspects:

**Five Core Components** (in `lmplay/base/runner/`):
1. **CurriculumComponent** - Manages training plans and current training stage
2. **ModelComponent** - Loads/saves model weights and construction parameters
3. **OptimizersComponent** - Creates optimizers with intelligent weight decay handling
4. **LRSchedulersComponent** - Manages learning rate schedules (warmup, etc.)
5. **RunnerComponent** - Tracks batch size, validation intervals, statistics

**Component Interface** (all components implement):
- `archive()` - Save current state: `{'state_args': {...}, 'state': {...}}`
- `advertise()` - Expose construction parameters for config files
- `construct(construction_args, state_args, state)` - Build/rebuild from merged config

**Construction Flow** in `runner.initialize()`:
1. Receive construction_args (from config), state_args_overrides (from CLI), checkpoint_state
2. Merge in order: construction_args → state_args_overrides → checkpoint_state
3. Construct components in order: curriculum → model → runner → optimizers → lr_schedulers
4. Components share access to the runner for coordination (e.g., model needed before optimizers)

**BasicModelRunner** is a simple implementation that:
- Takes a model class in `__init__`
- Implements `_construct_model()` to instantiate it
- Handles parameter overrides for different model versions
- Inherits all component orchestration from `LMRunnerBase`

### 3. Training Infrastructure

#### Datasets and Batching
- `lmplay/train/datasets/utils.py`: Handles loading, batching, and continuation of long texts
- **Key feature**: Intelligent text continuation at sentence boundaries
- Prevents information leakage by not continuing samples across batch boundaries

#### Training Loop
- `lmplay/train/__main__.py`: Main entry point
- Supports multi-stage training via **Training Plans**
- Each stage can use different datasets
- Resumes from interruption using checkpoint information

#### Statistics
- `lmplay/stats/modelstats.py`: Tracks loss, accuracy, tokens
- Per-step statistics for multi-stage training
- Exported to JSON for plotting

### 4. Modules

Reusable transformer components in `lmplay/modules/`:

- **blocks.py**: Transformer blocks with attention and feed-forward
- **attn.py**: Multi-head attention implementations
- **embeddings.py**: Embedding variations
- **weights.py**: Weight manipulation utilities (for experiments)

## Data Flow

### Training

```
1. User calls: lmp_trainer --exp gpt2ish --device cuda --amp
   ↓
2. train/__main__.py loads training plan and creates runner
   ↓
3. Runner.initialize() creates model and optimizer
   ↓
4. For each training stage:
     ├─ Load dataset via HuggingFace
     ├─ Create batcher for intelligent text chunking
     ├─ For each batch:
     │  ├─ Model.train_prompts() computes loss
     │  ├─ Runner handles gradient accumulation
     │  ├─ Optimizer.step() updates weights
     │  └─ Statistics tracked
     ├─ Periodic validation
     └─ Checkpoint saved
   ↓
5. Training complete
```

### Generation (Inference)

```
1. User calls: lmp_generator --model model.lmp --prompt "text"
   ↓
2. generate/__main__.py loads model and creates runner
   ↓
3. Runner.initialize(for_train=False)
   ↓
4. Model.generate_prompts() with KV-cache:
     ├─ Initialize cache: []
     ├─ For each generation step:
     │  ├─ Forward pass: (logits, cache) = forward(x, cache)
     │  ├─ Sample next token from logits
     │  └─ Append to cache for next iteration
     └─ Return decoded text
   ↓
5. Print results
```

## Experiment Pattern

Each experiment follows this structure:

```
lmplay/exp/your_experiment/
├── __init__.py              (empty or re-exports)
├── model.py                 (your model class)
└── versions.py              (runner definitions)
```

**Key pattern**:

```python
# model.py
class YourModel(LMBase):
    def __init__(self, ...):
        super().__init__(...)
        # Your architecture

    def forward(self, x, cache=None):
        # Your forward pass

# versions.py
@expose_runner('your_exp', description='...')
def runner(*args, **kwargs):
    return BasicModelRunner(YourModel, *args, **kwargs)
```

**Registration** (in `lmplay/__init__.py`):
```python
import lmplay.exp.your_experiment.versions
```

This makes it available as `lmp_trainer --exp your_exp`.

## Key Design Decisions

### 1. Copy-Paste Over Inheritance

Experiments are copied from baselines rather than using inheritance. This:
- Makes each experiment self-contained and independent
- Avoids complex inheritance hierarchies
- Allows easy modification without affecting other experiments
- Makes it clear what each experiment does

### 2. Runner Pattern

Models are wrapped in runners because:
- Separates model definition from training infrastructure
- Allows reusing training code across different model types
- Makes it easy to swap models without changing training logic
- Handles device management consistently

### 3. Gradient Accumulation Built-In

- Framework automatically handles `batch_size` vs `mini_batch_size`
- User specifies what they want (`batch_size`), framework handles GPU constraints (`mini_batch_size`)
- Simplifies experimentation on limited hardware

### 4. KV-Caching for Generation

- Models implement optional key-value caching
- Generation is slow but works - not optimized for speed
- Design priority: easy to understand and modify

### 5. Single GPU, No Distribution

- Simplifies reasoning about gradient accumulation
- No distributed training complexity
- Easier to debug and understand

## Extension Points

### Adding a New Model
1. Create `lmplay/exp/your_model/model.py` with a class extending `LMBase`
2. Create `lmplay/exp/your_model/versions.py` with `@expose_runner` decorators
3. Import in `lmplay/__init__.py`

### Adding New Dataset Support
1. Modify `lmplay/train/datasets/utils.py` batching logic
2. Update `lmplay/train/datasets/plan_configs.py` for new plans

### Adding New Training Features
1. Modify `lmplay/base/base_model.py` `LMRunnerBase` class
2. Or subclass `LMRunnerBase` for custom behavior

### External Experiments
1. Create separate Python package importing lmplay
2. Define experiments in your package following the same pattern
3. Register with lmplay's model registry via imports
4. Train using `lmp_trainer --exp your_experiment`

## Configuration and Hyperparameters

### Training Defaults (in `lmplay/base/base_model.py`)

```python
DEFAULT_LR = 6e-4
DEFAULT_WEIGHT_DECAY = 0.0
```

### Model Parameters (via CLI)

```bash
# Architecture
--num-blocks 6              # Number of transformer layers
--embed-dim 768             # Embedding dimension
--num-heads 12              # Number of attention heads
--context_len 1024          # Sequence length

# Training
--batch-size 50             # Effective batch size
--mini-batch-size 4         # GPU memory constraint
--lr 0.0006                 # Learning rate
--grad-clip 0.1             # Gradient clipping (optional)

# Optimization
--amp                       # Automatic Mixed Precision
--optimizer-warmup-fraction 0.1  # LR warmup fraction
```

See `lmplay/train/__main__.py` for full CLI options.