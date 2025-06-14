"""
Modular building blocks for transformer-based language models.

This module provides a collection of swappable, composable components that can be used
to build and experiment with different transformer architectures. The modules are designed
to be drop-in replacements for standard PyTorch components with additional experimental
features.

Key module categories:
- **blocks**: Transformer encoder blocks with configurable components
- **attn**: Multi-head attention implementations with experimental features
- **general**: General utility modules (residual connections, no-op modules)
- **embeddings**: Token and positional embedding implementations including Unified Embeddings
- **weights**: Experimental linear layer implementations with learned biases and sacrificial networks

The modules follow a consistent pattern where they can accept alternative implementations
of sub-components, allowing for easy experimentation with different architectures.
"""

from .blocks import *
from .attn import *
from .general import *
from .embeddings import *
#Unified Linear - Has mbias, mbias-bias, bias-bias
#Deep Unified Linear - Has a sacrificial network that predicts the bias and mbias
from .weights import *
