"""Base module for lmplay framework.

This module provides the foundational classes and utilities for building language models
and other neural network architectures within the lmplay framework. It includes:

- Base model classes (MBase, LMBase) that all models inherit from
- Recurrent model base classes (RMBase, RLMBase) for sequence modeling
- Runner system for training and inference management
- Model registration system for exposing models to the CLI

The base module follows a modular design pattern where models can be composed
from swappable components and experiments can easily extend the base functionality.
"""