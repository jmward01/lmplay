from lmplay.base.runner_list import MODEL_RUNNERS

#gotta force the classes to load so that the model runner list will populate with them. This needs to be somewhere so why not centralized here?
import lmplay.base.encoder.model
import lmplay.exp.weights.uw6.uw6_0.runner
import lmplay.exp.weights.uw6.uw6_1.runner
import lmplay.exp.weights.uw6.uw6_2.runner
import lmplay.exp.combined.sacrificial.sac2.sac2_0.runner
import lmplay.exp.combined.sacrificial.sac2.sac2_1.runner
import lmplay.exp.combined.sacrificial.sac2.sac2_2.runner



import lmplay.exp.embeddings.unified_embeddings_v1_0.model
import lmplay.exp.embeddings.unified_embeddings_v1_1.model
import lmplay.exp.combined.sacrificial.sac1_0.model
import lmplay.exp.combined.sacrificial.sac1_1.model
import lmplay.exp.attn_norm.normv.model

import lmplay.exp.weights.unified_weights_v1_0.model
import lmplay.exp.weights.unified_weights_v2_0.model
import lmplay.exp.weights.unified_weights_v2_1.model
