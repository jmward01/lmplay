from lmplay.exp.embeddings.unified_embeddings_v1_0.model import ModelRunner as UE8xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_0_1.model import ModelRunner as UE8x1ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_1.model import ModelRunner as UE16xModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_1_1.model import ModelRunner as UE16xGPUModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_2.model import ModelRunner as UE8x1_2ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_3.model import ModelRunner as UE8x1_3ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v1_4.model import ModelRunner as UE1_4ModelRunner
from lmplay.exp.embeddings.unified_embeddings_v2_0.model import ModelRunner as UE8x2_0ModelRunner
from lmplay.exp.combined.sacrificial.sac1_0.model import ModelRunner as SAC1_0ModelRunner
from lmplay.exp.combined.sacrificial.sac1_0_1.model import ModelRunner as SAC1_0_1ModelRunner
from lmplay.exp.combined.sacrificial.sac1_1.model import ModelRunner as SAC1_1ModelRunner
from lmplay.exp.combined.sacrificial.sac2.sac2_0.runner import ModelRunner as SAC2_0ModelRunner
from lmplay.exp.attn_norm.normv.model import ModelRunner as NormVModelRunner
from lmplay.base.encoder.model import ModelRunner as GPT2ishModelRunner
from lmplay.base.base_model import LMRunnerBase
from lmplay.exp.weights.unified_weights_v1_0.model import ModelRunner as UW1_0ModelRunner
from lmplay.exp.weights.unified_weights_v2_0.model import ModelRunner as UW2_0ModelRunner
from lmplay.exp.weights.unified_weights_v2_1.model import ModelRunner as UW2_1ModelRunner
from lmplay.exp.weights.unified_weights_v2_2.model import ModelRunner as UW2_2ModelRunner
from lmplay.exp.weights.unified_weights_v2_3.model import ModelRunner as UW2_3ModelRunner
from lmplay.exp.weights.unified_weights_v2_4.model import ModelRunner as UW2_4ModelRunner
from lmplay.exp.weights.unified_weights_v2_5.model import ModelRunner as UW2_5ModelRunner
from lmplay.exp.weights.unified_weights_v2_6.model import ModelRunner as UW2_6ModelRunner
from lmplay.exp.weights.unified_weights_v3_0.model import ModelRunner as UW3_0ModelRunner
from lmplay.exp.weights.unified_weights_v3_1.model import ModelRunner as UW3_1ModelRunner
from lmplay.exp.weights.unified_weights_v3_2.model import ModelRunner as UW3_2ModelRunner
from lmplay.exp.weights.unified_weights_v3_3.model import ModelRunner as UW3_3ModelRunner
from lmplay.exp.weights.unified_weights_v4_0.model import ModelRunner as UW4_0ModelRunner
from lmplay.exp.weights.unified_weights_v4_1.model import ModelRunner as UW4_1ModelRunner
from lmplay.exp.weights.unified_weights_v4_2.model import ModelRunner as UW4_2ModelRunner
from lmplay.exp.weights.unified_weights_v5_0.model import ModelRunner as UW5_0ModelRunner
from lmplay.exp.weights.unified_weights_v5_1.model import ModelRunner as UW5_1ModelRunner
from lmplay.exp.weights.unified_weights_v5_2.model import ModelRunner as UW5_2ModelRunner
from lmplay.exp.weights.uw6.uw6_0.runner import ModelRunner as UW6_0ModelRunner
from lmplay.exp.weights.uw6.uw6_1.runner import ModelRunner as UW6_1ModelRunner

MODEL_RUNNERS = dict(gpt2ish=GPT2ishModelRunner,
                     normv=NormVModelRunner,
                     sac1_0=SAC1_0ModelRunner,
                     sac16x1_0=SAC1_0_1ModelRunner,
                     sac16x1_1=SAC1_1ModelRunner,
                     sac2_0=SAC2_0ModelRunner,
                     ue8x=UE8xModelRunner,
                     ue8x1=UE8x1ModelRunner,
                     ue16x=UE16xModelRunner,
                     ue16xgpu=UE16xGPUModelRunner,
                     ue8x1_2=UE8x1_2ModelRunner,
                     ue8x1_3=UE8x1_3ModelRunner,
                     ue8x2_0=UE8x2_0ModelRunner,
                     ue1_4=UE1_4ModelRunner,
                     uw1_0=UW1_0ModelRunner,
                     uw2_0=UW2_0ModelRunner,
                     uw2_1=UW2_1ModelRunner,
                     uw2_2=UW2_2ModelRunner,
                     uw2_3=UW2_3ModelRunner,
                     uw2_4=UW2_4ModelRunner,
                     uw2_5=UW2_5ModelRunner,
                     uw2_6=UW2_6ModelRunner,
                     uw3_0=UW3_0ModelRunner,
                     uw3_1=UW3_1ModelRunner,
                     uw3_2=UW3_2ModelRunner,
                     uw3_3=UW3_3ModelRunner,
                     uw4_0=UW4_0ModelRunner,
                     uw4_1=UW4_1ModelRunner,
                     uw4_2=UW4_2ModelRunner,
                     uw5_0=UW5_0ModelRunner,
                     uw5_1=UW5_1ModelRunner,
                     uw5_2=UW5_2ModelRunner,
                     uw6_0=UW6_0ModelRunner,
                     uw6_1=UW6_1ModelRunner)

