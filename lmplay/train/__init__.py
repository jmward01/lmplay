"""
Training package for the lmplay language model experimentation framework.

This package provides the complete training pipeline including:
- Main training loop with multi-stage training plans
- Dataset loading and processing utilities  
- Integration with HuggingFace datasets
- Batch processing with automatic continuation for long texts
- Validation, checkpointing, and statistics tracking

The training system supports:
- Multi-stage training plans (pretraining -> finetuning)
- Gradient accumulation for effective larger batch sizes
- Automatic Mixed Precision (AMP) training
- Model compilation optimizations
- Optimizer warmup to prevent model shock
- Configurable validation intervals and save checkpoints
"""