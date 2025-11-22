"""
lmplay: A framework for rapid experimentation with language model architectures.

This package provides a modular system for building, training, and experimenting with
transformer-based language models. It features:

- Modular architecture with swappable components (attention, embeddings, blocks)
- Runner pattern for exposing models to CLI tools
- Multi-stage training plans with automatic dataset management
- Built-in experiment tracking and statistics
- Support for various experimental architectures

The MODEL_RUNNERS registry is populated by importing experiment version modules below,
which register their runners when loaded.
"""

from lmplay.base.runner_list import MODEL_RUNNERS

# Import experiment versions to register their runners in MODEL_RUNNERS.
# This enables the CLI tools to discover and use these experiments.

import lmplay.exp.weights.uw1.versions
import lmplay.exp.weights.uw2.versions
import lmplay.exp.weights.uw2.unified_weights_v2_1.model
import lmplay.exp.weights.uw6.versions

import lmplay.exp.combined.sacrificial.sac1.versions
import lmplay.exp.combined.sacrificial.sac2.versions


import lmplay.exp.embeddings.ue1_0.versions

import lmplay.exp.attn_norm.normv.versions

import lmplay.exp.nnmem.v1.versions
import lmplay.exp.nnmem.v2.versions
import lmplay.exp.nnmem.v3.versions
import lmplay.exp.nnmem.v4.versions

import lmplay.exp.attn_pos.v1.versions

import lmplay.exp.connections.lra.v1.versions

import lmplay.exp.combined.all.all1.versions

import lmplay.exp.gpt2ish.versions
import lmplay.exp.rgpt2ish.versions

import lmplay.exp.focus.v1.versions