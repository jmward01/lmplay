"""
Text generation module for language models.

This module provides text generation capabilities for trained language models,
accessible through the CLI tool 'lmp_generator'. It supports autoregressive
text generation with various configuration options.

The module integrates with the model runner system to provide a unified interface
for text generation across different experimental architectures.

Key features:
- Autoregressive text generation
- Support for all registered model experiments
- Configurable generation parameters
- Prompt loading from files or command line
- Device selection (CPU, CUDA, MPS)
- Automatic mixed precision support
"""