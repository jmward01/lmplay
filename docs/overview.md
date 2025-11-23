# LMPlay Documentation Overview

Welcome to the LMPlay documentation. This directory contains guides for understanding and working with the LMPlay language model experimentation framework.

## Quick Navigation

- **[Architecture](./architecture.md)** - High-level system design and how components interact
- **[Models & Runners](./models.md)** - Understanding the model abstraction and runner pattern
- **[Training System](./training.md)** - Multi-stage training, datasets, and validation
- **[GPT2ish Baseline](./gpt2ish_baseline.md)** - Reference implementation and baseline model

## Key Concepts

### The Framework

LMPlay is designed around **rapid experimentation** with language model architectures. The key principles are:

1. **Copy-paste-modify approach**: Experiments are intentionally copied from baselines rather than heavily shared, making each experiment independent and easy to understand
2. **Runner pattern**: Models are wrapped in runners that handle training logistics, checkpointing, and statistics
3. **Multi-stage training**: Support for curriculum learning and progressive training plans
4. **Simplified training**: Single-GPU training with gradient accumulation, designed to be easy to experiment with

### Experiments vs Core

The framework has two main parts:

- **Core** (`lmplay/base`, `lmplay/train`, `lmplay/generate`, etc.): Stable infrastructure for training, dataset handling, and statistics
- **Experiments** (`lmplay/exp`): Various model architectures and techniques being tested. These change frequently and are not backward compatible

## Getting Started

1. Read [Architecture](./architecture.md) to understand the overall design
2. Look at [GPT2ish Baseline](./gpt2ish_baseline.md) to see a complete model implementation
3. Review [Models & Runners](./models.md) to understand the abstraction layer
4. Check [Training System](./training.md) for how training, datasets, and validation work

## For Different Goals

### I want to understand how the code works
Start with [Architecture](./architecture.md), then read through the core model classes in `lmplay/base/base_model.py`.

### I want to create a new experiment
1. Read [Architecture](./architecture.md) and [GPT2ish Baseline](./gpt2ish_baseline.md)
2. Copy the gpt2ish experiment as a template
3. Modify the model implementation
4. Register it as a runner via `@expose_runner` decorator
5. Import it in `lmplay/__init__.py`

**Advanced**: If your experiments are complex or proprietary, you can create a separate Python package that imports lmplay as a library dependency. See "External Experiments" section below.

### I want to modify the training system
1. Read [Training System](./training.md)
2. Review the dataset handling in `lmplay/train/datasets/`
3. Look at the main training loop in `lmplay/train/__main__.py`

## External Experiments

LMPlay is designed to be imported and extended in external repositories. This allows you to develop experiments without modifying the core lmplay codebase.

### Setting Up an External Experiments Repository

1. **Create a new Python package** with its own project structure:
```
your_experiments/
├── src/your_experiments/
│   ├── __init__.py
│   └── exp/
│       ├── __init__.py
│       └── your_experiment/
│           ├── __init__.py
│           ├── model.py
│           └── versions.py
├── pyproject.toml
└── README.md
```

2. **Install lmplay as a dependency** in your `pyproject.toml`:
```toml
[project]
dependencies = [
    "lmplay @ file:///path/to/lmplay",
]
```

3. **Create experiments following the same pattern** as lmplay experiments:
```python
# src/your_experiments/exp/your_experiment/model.py
from lmplay.base.base_model import LMBase
# Your model implementation

# src/your_experiments/exp/your_experiment/versions.py
from lmplay.base.runner_list import expose_runner
from lmplay.base.runners import BasicModelRunner
from .model import YourModel

@expose_runner('your_experiment', description='Your experiment description')
def runner(*args, **kwargs):
    return BasicModelRunner(YourModel, *args, **kwargs)
```

4. **Register your experiments** in `src/your_experiments/__init__.py`:
```python
from lmplay import MODEL_RUNNERS
import your_experiments.exp.your_experiment.versions
```

5. **Install your package** in development mode:
```bash
pip install -e /path/to/your_experiments
```

6. **Run training** using your registered experiments:
```bash
lmp_trainer --exp your_experiment --device cuda --amp
```

### Benefits of External Repositories

- **Separation of concerns**: Core framework stays stable, experiments stay flexible
- **No core modifications needed**: Develop freely without worrying about breaking lmplay
- **Easy collaboration**: Share just your experiments package, not a fork of lmplay
- **Version control**: Experiments and lmplay can be versioned independently

## Common Commands

```bash
# List available experiments
lmp_trainer --exp list

# Describe an experiment
lmp_trainer --exp ue1_0 --describe

# Train the baseline model
lmp_trainer --device cuda --amp

# Train a specific experiment
lmp_trainer --device cuda --amp --exp ue1_0

# Generate text from a trained model
lmp_generator --model model_name.lmp --prompt "Start text"

# Create comparison plots of training runs
lmp_plotstats
```

## File Structure

```
lmplay/
├── base/                 # Core model and runner abstractions
├── exp/                  # Experiments (each with own architecture)
├── modules/              # Reusable transformer components
├── train/                # Training loop and dataset handling
├── generate/             # Inference and text generation
├── stats/                # Statistics tracking and visualization
├── cleanmodel/           # Utilities for production model export
└── utils.py              # Shared utilities
```

## Resources

- **README.md**: High-level project overview and experiment descriptions
- **Source code**: Core abstractions well-documented with docstrings

For specific implementation details, always check the source code first - docstrings are the authoritative reference.