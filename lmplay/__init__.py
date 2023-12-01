from lmplay.exp.embeddings.unified_embeddings_v1_0.model import ModelRunner as UE8xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_0_1.model import ModelRunner as UE8x1ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_1.model import ModelRunner as UE16xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_2.model import ModelRunner as UE8x1_2ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v2_0.model import ModelRunner as UE8x2_0ModelRunner
from lmplay.exp.attn_norm.normv.model import ModelRunner as NormVModelRunner
from lmplay.base.encoder.model import ModelRunner as GPT2ishModelRunner
from lmplay.base.base_model import LMRunnerBase


MODEL_RUNNERS = dict(gpt2ish=GPT2ishModelRunner,
                     normv=NormVModelRunner,
                     ue8x=UE8xModelRunner,
                     ue8x1_2=UE8x1_2ModelRunner,
                     ue8x1=UE8x1ModelRunner,
                     ue8x2_0=UE8x2_0ModelRunner,
                     ue16x=UE16xModelRunner)

