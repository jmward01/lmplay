from lmplay.exp.embeddings.unified_embeddings_v1_0.model import ModelRunner as UE8xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_0_1.model import ModelRunner as UE8x1ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_1.model import ModelRunner as UE16xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_1_1.model import ModelRunner as UE16xGPUModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_2.model import ModelRunner as UE8x1_2ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_3.model import ModelRunner as UE8x1_3ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v2_0.model import ModelRunner as UE8x2_0ModelRunner
from lmplay.exp.combined.sacrificial.sac1_0.model import ModelRunner as SAC1_0ModelRunner
from lmplay.exp.combined.sacrificial.sac1_0_1.model import ModelRunner as SAC1_0_1ModelRunner
from lmplay.exp.combined.sacrificial.sac1_1.model import ModelRunner as SAC1_1ModelRunner
from lmplay.exp.attn_norm.normv.model import ModelRunner as NormVModelRunner
from lmplay.base.encoder.model import ModelRunner as GPT2ishModelRunner
from lmplay.base.base_model import LMRunnerBase
from lmplay.exp.weights.unified_weights_v1_0.model import ModelRunner as UW1_0ModelRunner
from lmplay.exp.weights.unified_weights_v2_0.model import ModelRunner as UW2_0ModelRunner
from lmplay.exp.weights.unified_weights_v2_1.model import ModelRunner as UW2_1ModelRunner

MODEL_RUNNERS = dict(gpt2ish=GPT2ishModelRunner,
                     normv=NormVModelRunner,
                     sac1_0=SAC1_0ModelRunner,
                     sac16x1_0=SAC1_0_1ModelRunner,
                     sac16x1_1=SAC1_1ModelRunner,
                     ue8x=UE8xModelRunner,
                     ue8x1=UE8x1ModelRunner,
                     ue16x=UE16xModelRunner,
                     ue16xgpu=UE16xGPUModelRunner,
                     ue8x1_2=UE8x1_2ModelRunner,
                     ue8x1_3=UE8x1_3ModelRunner,
                     ue8x2_0=UE8x2_0ModelRunner,
                     uw1_0=UW1_0ModelRunner,
                     uw2_0=UW2_0ModelRunner,
                     uw2_1=UW2_1ModelRunner)

