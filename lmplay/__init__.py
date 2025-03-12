from lmplay.base.runner_list import MODEL_RUNNERS

#gotta force the classes to load so that the model runner list will populate with them. This needs to be somewhere so why not centralized here?

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