from .blocks import *
from .attn import *
from .general import *
from lmplay.exp.embeddings.modules import UnifiedEmbedding, ConvertableEmbedding
#Unified Linear - Has mbias, mbias-bias, bias-bias
#Deep Unified Linear - Has a sacrificial network that predicts the bias and mbias
from lmplay.exp.weights.modules import ULinear , DULinear, SDULinear, SPredictor, SimpleMLP, MultiMLP, accepts_purpose
