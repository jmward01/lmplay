from lmplay.base.encoder.modules import MultiheadAttention, Block
from lmplay.exp.embeddings.unified_embeddings_v1_0.modules import UnifiedEmbedding, ConvertableEmbedding
#Unified Linear - Has mbias, mbias-bias, bias-bias
from lmplay.exp.weights.unified_weights_v1_0.modules import ULinear
#Deep Unified Linear - Has a sacrificial network that predicts the bias
from lmplay.exp.weights.unified_weights_v3_0.modules import DULinear
