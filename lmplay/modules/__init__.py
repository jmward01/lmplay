from lmplay.base.encoder.modules import MultiheadAttention, Block
from lmplay.exp.embeddings.unified_embeddings_v1_0.modules import UnifiedEmbedding, ConvertableEmbedding
#Unified Linear - Has mbias, mbias-bias, bias-bias
#Deep Unified Linear - Has a sacrificial network that predicts the bias and mbias
from lmplay.exp.weights.modules import ULinear #, DULinear
from lmplay.exp.weights.unified_weights_v5_0.modules import DULinear
