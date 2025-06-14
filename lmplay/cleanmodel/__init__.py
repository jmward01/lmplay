"""
Model cleaning and preparation utilities.

This module provides functionality to clean and prepare trained models for
deployment by removing training-specific parameters and optimizing the model
structure for inference.

The cleaning process typically involves:
- Removing sacrificial training parameters (like UE integration layers)
- Converting complex training architectures to simple inference forms
- Optimizing memory usage and loading speed
- Preparing models for production deployment

The module is accessible through the CLI tool 'lmp_cleanmodel' for easy
model preparation workflows.
"""